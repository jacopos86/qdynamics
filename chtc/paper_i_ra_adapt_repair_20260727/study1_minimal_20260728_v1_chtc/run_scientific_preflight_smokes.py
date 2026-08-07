#!/usr/bin/env python3
"""Run the actual P2/P3 scientific preflight and emit one signed receipt.

P2 reconstructs the historical nph=3/7 characterization inventories from
the public pool builders.  P3 executes four real nph=3 representation smokes:
RA plateau with macro and guarded-singleton candidates, and conventional
Append-ADAPT with each representation.  Each RA smoke exercises a bounded
fresh run, an authenticated 2->3 continuation, an independent three-round
run, and a separately labelled authenticated continuation to its locked first
interior-position round.  The short RA legs are chart/replay evidence only and
must fail aggregate G5 for exactly one reason: no interior scored receipt yet.
The separate witness must pass strict aggregate G5.  Append-ADAPT exercises
two independent runs and its explicit reconstruction-only checkpoint boundary.

The script never mutates the v8 materialization and refuses to overwrite its
output receipt.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from objective_gates import _validate_g10, _validate_g5
from package_contract import (
    PACKAGE_ID,
    V8_RELATIVE_ROOT,
    PackageContractError,
    canonical_json_bytes,
    canonical_sha256,
    digested,
    package_control_plane_receipt,
    repo_root_from_script,
    sha256_file,
    validate_v8_authority,
)


PREFLIGHT_SCHEMA = "paper_i_ra_adapt_study1_scientific_preflight_v3"
P2_SCHEMA = "paper_i_ra_adapt_study1_p2_pool_smoke_v2"
P3_SCHEMA = "paper_i_ra_adapt_study1_p3_semantic_smoke_v3"
EXACT_INSERTION_CHART = "exact_ordered_insertion_zero_angle_v1"
G5_MODE_REQUIRED = "strict_aggregate_required_v1"
G5_MODE_SHORT_PREFIX = "short_prefix_interior_witness_deferred_v1"

# These are measured protocol-specific bounds, not tunable search limits.  P3
# fails closed if an interior record appears earlier or does not first appear
# at the locked bound.
G5_FIRST_INTERIOR_ROUND_BY_REPRESENTATION = {
    "macro_generator_v1": 13,
    "single_pauli_word_v1": 13,
}

POOL_EXPECTATIONS = {
    3: {
        "parent_count": 123,
        "parent_labels_sha256": (
            "17cc97b744f8e6b50b686b24edd28426ca2c055bc2c31054fd353ddfa10efbe3"
        ),
        "parent_pool_sha256": (
            "b533c4e08e57683bfb42de7a811ef106ba0eaa94f75d9a57f907cc36370fa67d"
        ),
        "macro_count": 102,
        "macro_labels_sha256": (
            "a8831528590e870a09ce08492b6f61da4a4d377e63fa8983b30ca9698af5d3d9"
        ),
        "macro_pool_sha256": (
            "1549f2e108406f494c2d4f884212c1026dbaa42f12eb92189f18eaf2a62b17df"
        ),
        "guarded_count": 948,
        "guarded_labels_sha256": (
            "02995a2c570d4322e46e55e3a532381ff7eff85dc3c2de8cb2b30ed888b76906"
        ),
        "guarded_pool_sha256": (
            "66ea9d6b058b562ba913e221124285de7be0ec13a972d30b9812ab29874a58e0"
        ),
    },
    7: {
        "parent_count": 171,
        "parent_labels_sha256": (
            "389ce1382b57b916e15e170c641f3884ed1ce33e9913d6eb709f24490739e93f"
        ),
        "parent_pool_sha256": (
            "831817f5a6a072ad2a43f4413b34fa6da558120081bbebce2831261ac03d680e"
        ),
        "macro_count": 148,
        "macro_labels_sha256": (
            "e6de937476653868f7d3974ad67c467c2f2e2496770e256671b2e807a5b5b03a"
        ),
        "macro_pool_sha256": (
            "e30e879dabf4d6eb234be92aae1cea76998172b67e8c679b241f5cdc6641d14e"
        ),
        "guarded_count": 6508,
        "guarded_labels_sha256": (
            "079478057eea213139dc2f3c7486097496454421a44677c290b5dc55860accb7"
        ),
        "guarded_pool_sha256": (
            "8e1fe54be4b089d759d334399add40fc5edea8faa31af9ea70f1f2cc36834e93"
        ),
    },
}


class PreflightError(RuntimeError):
    """Raised when an actual smoke fails its objective contract."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreflightError(f"{label} is not a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise PreflightError(f"{label} is not a sequence.")
    return value


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PreflightError(f"{label} is unavailable or symlinked: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PreflightError(f"{label} must contain a JSON object.")
    return payload


def _assert_checkpoint_signatures(
    payload: Mapping[str, Any],
    *,
    label: str,
    append: bool,
    expected_rounds: int,
) -> str:
    """Validate the method-specific checkpoint signature boundary.

    Append signs the complete outer observation with ``sha256``.  RA's
    resumable outer envelope is authenticated as a transported file and
    contains a chain of signed active-prefix checkpoints whose signature field
    is ``checkpoint_sha256``.  Treating those two formats as interchangeable
    would either reject valid RA checkpoints or skip their actual signed
    scientific boundary.
    """

    if append:
        if (
            payload.get("schema")
            != "paper_i_append_adapt_checkpoint_v1"
            or "checkpoint_sha256" in payload
        ):
            raise PreflightError(f"{label} Append schema/signature drifted.")
        observed = payload.get("sha256")
        unsigned = dict(payload)
        unsigned.pop("sha256", None)
        expected = canonical_sha256(unsigned)
        if observed != expected:
            raise PreflightError(f"{label} self digest does not match.")
        return expected

    if (
        payload.get("schema_version")
        != "static_adapt_current_checkpoint_v1"
        or "sha256" in payload
        or "checkpoint_sha256" in payload
    ):
        raise PreflightError(f"{label} RA outer checkpoint schema drifted.")
    checkpoint = _mapping(
        payload.get("checkpoint"), label=f"{label} checkpoint envelope"
    )
    adapt = _mapping(
        payload.get("adapt_vqe"), label=f"{label} ADAPT payload"
    )
    prefixes = _sequence(
        adapt.get("active_prefix_checkpoints"),
        label=f"{label} active-prefix checkpoints",
    )
    terminal = _mapping(
        adapt.get("terminal_active_prefix_checkpoint"),
        label=f"{label} terminal active-prefix checkpoint",
    )
    continuation = _mapping(
        adapt.get("continuation"),
        label=f"{label} continuation payload",
    )
    if (
        checkpoint.get("depth") != expected_rounds
        or adapt.get("history_count") != expected_rounds
        or len(prefixes) != expected_rounds
        or terminal.get("outer_iteration") != expected_rounds
        or continuation.get("terminal_active_prefix_checkpoint") != terminal
    ):
        raise PreflightError(f"{label} RA checkpoint depth chain drifted.")
    signed_rows = [*prefixes, terminal]
    terminal_digest = ""
    for row_index, raw_prefix in enumerate(signed_rows):
        ordinal = (
            row_index + 1
            if row_index < len(prefixes)
            else expected_rounds
        )
        prefix = _mapping(
            raw_prefix,
            label=(
                f"{label} active prefix {ordinal}"
                if row_index < len(prefixes)
                else f"{label} terminal active prefix"
            ),
        )
        if (
            prefix.get("schema")
            != "paper_i_signed_active_prefix_checkpoint_v1"
            or prefix.get("outer_iteration") != ordinal
            or "sha256" in prefix
        ):
            raise PreflightError(
                f"{label} active-prefix schema/order drifted."
            )
        observed = prefix.get("checkpoint_sha256")
        unsigned = dict(prefix)
        unsigned.pop("checkpoint_sha256", None)
        expected = canonical_sha256(unsigned)
        if observed != expected:
            raise PreflightError(
                f"{label} active-prefix checkpoint digest does not match."
            )
        if row_index == len(prefixes):
            terminal_digest = expected
    return terminal_digest


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload) + b"\n"
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def _temporary_environment(
    updates: Mapping[str, str],
) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update({name: str(value) for name, value in updates.items()})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _immutable_tree_snapshot(root: Path) -> dict[str, Any]:
    """Capture content plus write-sensitive stat identity, excluding atime."""

    resolved = root.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    for path in (resolved, *sorted(resolved.rglob("*"))):
        relative = "." if path == resolved else path.relative_to(resolved).as_posix()
        if path.is_symlink():
            raise PreflightError(
                f"Immutable v8 authority contains a symlink: {relative}"
            )
        stat = path.stat()
        if path.is_dir():
            kind = "directory"
            content_sha256 = None
        elif path.is_file():
            kind = "file"
            content_sha256 = sha256_file(path)
        else:
            raise PreflightError(
                f"Immutable v8 authority has an unsupported member: {relative}"
            )
        rows.append(
            {
                "path": relative,
                "kind": kind,
                "mode": stat.st_mode,
                "uid": stat.st_uid,
                "gid": stat.st_gid,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "nlink": stat.st_nlink,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
                "content_sha256": content_sha256,
            }
        )
    return {
        "schema": "paper_i_immutable_tree_stat_snapshot_v1",
        "entry_count": len(rows),
        "rows_sha256": canonical_sha256(rows),
        "rows": rows,
    }


def _fixture_problem(*, nph: int) -> Any:
    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import (
        resolve_problem_context,
    )

    return resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.5,
            dv=0.0,
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=nph,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        )
    )


def _run_p2() -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt.contracts import (
        CANDIDATE_REPRESENTATION_MACRO,
        CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    )
    from pipelines.static_adapt.ra_adapt.pools import (
        build_candidate_inventory_lineage_receipt,
        build_executable_macro_pool,
        build_guarded_single_pauli_pool,
        build_parent_template_inventory,
    )

    rows: list[dict[str, Any]] = []
    for nph in (3, 7):
        problem = _fixture_problem(nph=nph)
        macro_parent = build_parent_template_inventory(
            problem,
            representation_id=CANDIDATE_REPRESENTATION_MACRO,
        )
        singleton_parent = build_parent_template_inventory(
            problem,
            representation_id=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        )
        macro = build_executable_macro_pool(problem)
        guarded = build_guarded_single_pauli_pool(problem)
        guarded_lineage = build_candidate_inventory_lineage_receipt(guarded)
        expected = POOL_EXPECTATIONS[nph]
        observed = {
            "parent_count": macro_parent.receipt.count,
            "parent_labels_sha256": (
                macro_parent.receipt.ordered_labels_sha256
            ),
            "parent_pool_sha256": macro_parent.receipt.ordered_pool_sha256,
            "macro_count": macro.receipt.count,
            "macro_labels_sha256": macro.receipt.ordered_labels_sha256,
            "macro_pool_sha256": macro.receipt.ordered_pool_sha256,
            "guarded_count": guarded.receipt.count,
            "guarded_labels_sha256": guarded.receipt.ordered_labels_sha256,
            "guarded_pool_sha256": guarded.receipt.ordered_pool_sha256,
        }
        if observed != expected:
            raise PreflightError(
                f"P2 nph={nph} inventory identity drifted: {observed!r}"
            )
        if (
            singleton_parent.receipt.count != expected["parent_count"]
            or singleton_parent.receipt.ordered_labels_sha256
            != macro_parent.receipt.ordered_labels_sha256
            or singleton_parent.receipt.ordered_pool_sha256
            != macro_parent.receipt.ordered_pool_sha256
            or guarded.metadata.get("exposure_scope")
            != "global_parent_inventory_v1"
            or guarded.metadata.get("source_parent_count")
            != expected["parent_count"]
        ):
            raise PreflightError(
                f"P2 nph={nph} singleton ancestry/exposure drifted."
            )
        rows.append(
            {
                "nph": nph,
                **observed,
                "macro_parent_receipt_sha256": (
                    macro_parent.receipt.sha256
                ),
                "singleton_parent_receipt_sha256": (
                    singleton_parent.receipt.sha256
                ),
                "parent_identity_equal_across_representations": True,
                "guarded_pool_receipt_sha256": guarded.receipt.sha256,
                "guarded_lineage_receipt_sha256": (
                    guarded_lineage.sha256
                ),
                "guarded_construction_manifest_sha256": canonical_sha256(
                    guarded.metadata["shared_pool_manifest"]
                ),
                "global_guarded_child_construction_passed": True,
            }
        )
    return digested(
        {
            "schema": P2_SCHEMA,
            "status": "passed",
            "actual_builder_execution": True,
            "rows": rows,
            "all_expected_identities_equal": True,
        }
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
    if ResolvedProblemReceipt.from_problem(problem).to_dict() != (
        receipt.to_dict()
    ):
        raise PreflightError(
            "P3 reconstructed problem does not match protocol authority."
        )
    return problem


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
    expected_rounds: int,
) -> Mapping[str, Any]:
    from pipelines.static_adapt.ra_adapt.replay_evidence import (
        validate_controller_replay_evidence,
    )

    scientific = _mapping(
        payload.get("scientific_receipts"),
        label="scientific receipts",
    )
    replay = validate_controller_replay_evidence(
        _mapping(
            scientific.get("controller_replay_evidence"),
            label="controller replay evidence",
        )
    )
    if (
        replay["sha256"]
        != scientific.get("controller_replay_evidence_sha256")
        or len(replay["signed_controller_round_prefixes"])
        != expected_rounds
        or len(_trajectory(payload, append=append)) != expected_rounds
    ):
        raise PreflightError(
            "P3 signed replay evidence does not close at its bound."
        )
    return replay


def _validate_g5_for_preflight(
    *,
    job: Mapping[str, Any],
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Run the production G5 validator without overstating short prefixes."""

    if mode == G5_MODE_REQUIRED:
        return {
            "mode": mode,
            "aggregate_g5_passed": True,
            "evidence": _validate_g5(
                job=job,
                protocol=protocol,
                result=result,
            ),
        }
    if mode != G5_MODE_SHORT_PREFIX:
        raise PreflightError(f"Unsupported P3 G5 validation mode: {mode!r}.")
    if job["execution_entrypoint"] != "run_ra_adapt":
        raise PreflightError(
            "Only RA short-prefix smokes may defer the interior witness."
        )
    expected = (
        f"G5 requires an interior scored receipt for "
        f"{job['execution_id']}."
    )
    try:
        _validate_g5(job=job, protocol=protocol, result=result)
    except PackageContractError as exc:
        if str(exc) != expected:
            raise PreflightError(
                f"{job['execution_id']} failed G5 before the intended "
                f"short-prefix interior boundary: {exc}"
            ) from exc
    else:
        raise PreflightError(
            f"{job['execution_id']} unexpectedly satisfied aggregate G5 "
            "inside the locked short-prefix bound."
        )
    return {
        "mode": mode,
        "aggregate_g5_passed": False,
        "strict_validator_outcome": "expected_missing_interior_only",
        "deferred_witness_role": "separate_g5_plateau_witness",
        "expected_error": expected,
    }


def _first_interior_witness_receipt(
    *,
    case_id: str,
    payload: Mapping[str, Any],
    expected_round: int,
) -> dict[str, Any]:
    scientific = _mapping(
        payload.get("scientific_receipts"),
        label=f"{case_id} scientific receipts",
    )
    accepted_rounds = _sequence(
        scientific.get("accepted_round_receipts"),
        label=f"{case_id} accepted-round receipts",
    )
    if len(accepted_rounds) != expected_round:
        raise PreflightError(
            f"{case_id} G5 witness closed at {len(accepted_rounds)} rounds; "
            f"expected {expected_round}."
        )
    per_round: list[dict[str, int]] = []
    for round_ordinal, raw_round in enumerate(accepted_rounds, start=1):
        accepted = _mapping(
            raw_round,
            label=f"{case_id} accepted round {round_ordinal}",
        )
        population = _mapping(
            accepted.get("scored_insertion_position_population"),
            label=f"{case_id} scored population {round_ordinal}",
        )
        observed_ordinal = accepted.get("accepted_round_ordinal")
        interior = population.get("interior_scored_count")
        if (
            isinstance(observed_ordinal, bool)
            or not isinstance(observed_ordinal, int)
            or observed_ordinal != round_ordinal
            or isinstance(interior, bool)
            or not isinstance(interior, int)
            or interior < 0
        ):
            raise PreflightError(
                f"{case_id} G5 witness population order/count drifted."
            )
        per_round.append(
            {
                "accepted_round_ordinal": round_ordinal,
                "interior_scored_count": interior,
            }
        )
    interior_rounds = [
        row["accepted_round_ordinal"]
        for row in per_round
        if row["interior_scored_count"] > 0
    ]
    if not interior_rounds or interior_rounds[0] != expected_round:
        raise PreflightError(
            f"{case_id} first interior round is "
            f"{interior_rounds[0] if interior_rounds else None}; "
            f"expected {expected_round}."
        )
    if interior_rounds != [expected_round]:
        raise PreflightError(
            f"{case_id} has an interior record before its locked first "
            "witness round."
        )
    return digested(
        {
            "schema": "paper_i_ra_adapt_g5_first_interior_witness_v1",
            "case_id": case_id,
            "expected_first_interior_round": expected_round,
            "observed_first_interior_round": interior_rounds[0],
            "rounds_before_witness_have_zero_interior": True,
            "witness_round_interior_scored_count": per_round[-1][
                "interior_scored_count"
            ],
            "per_round_interior_counts": per_round,
            "status": "passed",
        }
    )


def _validate_observed_run(
    *,
    case_id: str,
    route_id: str,
    payload: Mapping[str, Any],
    checkpoint_path: Path,
    ledger_path: Path,
    append: bool,
    expected_rounds: int,
    g5_mode: str = G5_MODE_REQUIRED,
) -> dict[str, Any]:
    protocol = _mapping(payload.get("protocol"), label=f"{case_id} protocol")
    checkpoint = _load_json(
        checkpoint_path, label=f"{case_id} checkpoint"
    )
    ledger = _load_json(ledger_path, label=f"{case_id} estimator ledger")
    checkpoint_signed_payload_sha256 = _assert_checkpoint_signatures(
        checkpoint,
        label=f"{case_id} checkpoint",
        append=append,
        expected_rounds=expected_rounds,
    )
    replay = _validated_replay(
        payload,
        append=append,
        expected_rounds=expected_rounds,
    )
    job = {
        "execution_id": case_id,
        "execution_entrypoint": (
            "run_append_adapt" if append else "run_ra_adapt"
        ),
        "route_id": route_id,
    }
    g5 = _validate_g5_for_preflight(
        job=job,
        protocol=protocol,
        result=payload,
        mode=g5_mode,
    )
    g10 = _validate_g10(job=job, result=payload, ledger=ledger)
    if protocol.get("derivative_chart_id") != EXACT_INSERTION_CHART:
        raise PreflightError(f"{case_id} exact insertion chart drifted.")

    resume = _mapping(
        replay.get("resume_sidecar_closure"),
        label=f"{case_id} resume closure",
    )
    if append:
        if (
            resume.get("resume_mode")
            != "authenticated_reconstruction_only_v1"
            or resume.get("public_resume_execution_supported") is not False
            or resume.get("reconstruction_fields_complete") is not True
            or checkpoint.get("schema")
            != "paper_i_append_adapt_checkpoint_v1"
            or checkpoint.get("controller_replay_evidence") != replay
        ):
            raise PreflightError(
                f"{case_id} Append reconstruction boundary drifted."
            )
    else:
        checkpoint_artifact = _mapping(
            resume.get("checkpoint_artifact"),
            label=f"{case_id} checkpoint artifact",
        )
        ledger_artifact = _mapping(
            resume.get("estimator_ledger_artifact"),
            label=f"{case_id} ledger artifact",
        )
        if (
            resume.get("resume_mode")
            != "canonical_accepted_state_resume_v1"
            or resume.get("public_resume_execution_supported") is not True
            or resume.get("authentication_binding_complete") is not True
            or checkpoint_artifact.get("sha256")
            != sha256_file(checkpoint_path)
            or checkpoint_artifact.get("size_bytes")
            != checkpoint_path.stat().st_size
            or ledger_artifact.get("sha256") != sha256_file(ledger_path)
            or ledger_artifact.get("size_bytes")
            != ledger_path.stat().st_size
            or _mapping(
                resume.get("estimator_prefix_closure"),
                label=f"{case_id} estimator prefix closure",
            ).get("passed")
            is not True
        ):
            raise PreflightError(f"{case_id} RA sidecar closure drifted.")
    return {
        "protocol_sha256": protocol["sha256"],
        "problem_request_sha256": protocol["problem"][
            "problem_request_sha256"
        ],
        "controller_rounds_completed": expected_rounds,
        "trajectory_sha256": canonical_sha256(
            _trajectory(payload, append=append)
        ),
        "controller_replay_evidence_sha256": replay["sha256"],
        "checkpoint_signed_payload_sha256": (
            checkpoint_signed_payload_sha256
        ),
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "estimator_ledger_file_sha256": sha256_file(ledger_path),
        "estimator_ledger_size_bytes": ledger_path.stat().st_size,
        "g5": g5,
        "g10": g10,
    }


def _run_ra_case(
    *,
    case_id: str,
    route_id: str,
    protocol_path: Path,
    work_root: Path,
) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt import (
        RAAdaptOperationalControls,
        run_ra_adapt,
    )
    from pipelines.static_adapt.ra_adapt.bundles import (
        load_validated_bundle_protocol,
    )
    from pipelines.static_adapt.ra_adapt.replay_evidence import (
        compare_bounded_controller_replays,
    )
    from pipelines.static_adapt.sr_snake import AcceptedStateResume

    protocol = load_validated_bundle_protocol(protocol_path)
    problem = _problem_from_protocol(protocol)
    witness_round = G5_FIRST_INTERIOR_ROUND_BY_REPRESENTATION.get(
        str(protocol.candidate_representation)
    )
    if (
        int(protocol.problem.n_ph_max) != 3
        or witness_round is None
        or int(protocol.horizon) < witness_round
    ):
        raise PreflightError(f"{case_id} is not an authorized nph=3 horizon.")
    case_root = work_root / case_id
    case_root.mkdir()
    bundle_root = protocol_path.parent.parent

    payloads: dict[str, Mapping[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}

    def capture_and_validate(
        *,
        stem: str,
        result: Any,
        expected_rounds: int,
        g5_mode: str,
    ) -> None:
        payload = result.to_dict()
        payloads[stem] = payload
        validations[stem] = _validate_observed_run(
            case_id=f"{case_id}__{stem}",
            route_id=route_id,
            payload=payload,
            checkpoint_path=case_root / f"{stem}.checkpoint.json",
            ledger_path=case_root / f"{stem}.ledger.json",
            append=False,
            expected_rounds=expected_rounds,
            g5_mode=g5_mode,
        )

    with _working_directory(bundle_root):
        for stem, rounds in (("primary", 3), ("fresh_leg", 2)):
            result = run_ra_adapt(
                problem,
                protocol,
                operational_controls=RAAdaptOperationalControls(
                    maximum_controller_rounds=rounds,
                    observation=_observation(case_root, stem=stem),
                ),
            )
            capture_and_validate(
                stem=stem,
                result=result,
                expected_rounds=rounds,
                g5_mode=G5_MODE_SHORT_PREFIX,
            )
        fresh_checkpoint = case_root / "fresh_leg.checkpoint.json"
        resumed = run_ra_adapt(
            problem,
            protocol,
            operational_controls=RAAdaptOperationalControls(
                maximum_controller_rounds=3,
                resume=AcceptedStateResume(
                    checkpoint_path=fresh_checkpoint,
                    checkpoint_sha256=sha256_file(fresh_checkpoint),
                ),
                observation=_observation(case_root, stem="resumed"),
            ),
        )
        capture_and_validate(
            stem="resumed",
            result=resumed,
            expected_rounds=3,
            g5_mode=G5_MODE_SHORT_PREFIX,
        )
        primary_checkpoint = case_root / "primary.checkpoint.json"
        witness = run_ra_adapt(
            problem,
            protocol,
            operational_controls=RAAdaptOperationalControls(
                maximum_controller_rounds=witness_round,
                resume=AcceptedStateResume(
                    checkpoint_path=primary_checkpoint,
                    checkpoint_sha256=sha256_file(primary_checkpoint),
                ),
                observation=_observation(
                    case_root,
                    stem="g5_plateau_witness",
                ),
            ),
        )
        capture_and_validate(
            stem="g5_plateau_witness",
            result=witness,
            expected_rounds=witness_round,
            g5_mode=G5_MODE_REQUIRED,
        )

    primary_trajectory = _trajectory(payloads["primary"], append=False)
    fresh_trajectory = _trajectory(payloads["fresh_leg"], append=False)
    resumed_trajectory = _trajectory(payloads["resumed"], append=False)
    witness_trajectory = _trajectory(
        payloads["g5_plateau_witness"],
        append=False,
    )
    if (
        primary_trajectory[:2] != fresh_trajectory
        or resumed_trajectory[:2] != fresh_trajectory
        or primary_trajectory != resumed_trajectory
        or witness_trajectory[:3] != primary_trajectory
    ):
        raise PreflightError(
            f"{case_id} bounded replay/resume trajectory identity failed."
        )
    primary_replay = _validated_replay(
        payloads["primary"], append=False, expected_rounds=3
    )
    fresh_replay = _validated_replay(
        payloads["fresh_leg"], append=False, expected_rounds=2
    )
    resumed_replay = _validated_replay(
        payloads["resumed"], append=False, expected_rounds=3
    )
    witness_replay = _validated_replay(
        payloads["g5_plateau_witness"],
        append=False,
        expected_rounds=witness_round,
    )
    fresh_comparison = compare_bounded_controller_replays(
        primary_replay, fresh_replay, controller_round=2
    )
    resume_comparison = compare_bounded_controller_replays(
        fresh_replay, resumed_replay, controller_round=2
    )
    witness_comparison = compare_bounded_controller_replays(
        primary_replay,
        witness_replay,
        controller_round=3,
    )
    if (
        fresh_comparison.get("matched") is not True
        or resume_comparison.get("matched") is not True
        or witness_comparison.get("matched") is not True
    ):
        raise PreflightError(
            f"{case_id} signed bounded replay comparison failed."
        )
    witness_receipt = _first_interior_witness_receipt(
        case_id=f"{case_id}__g5_plateau_witness",
        payload=payloads["g5_plateau_witness"],
        expected_round=witness_round,
    )
    witness_g5 = _mapping(
        validations["g5_plateau_witness"].get("g5"),
        label=f"{case_id} witness G5",
    )
    witness_g5_evidence = _mapping(
        witness_g5.get("evidence"),
        label=f"{case_id} witness G5 evidence",
    )
    if (
        witness_g5.get("aggregate_g5_passed") is not True
        or witness_g5_evidence.get("interior_scored_count", 0) < 1
    ):
        raise PreflightError(f"{case_id} strict G5 witness did not close.")
    return {
        "case_id": case_id,
        "method_family": "ra_adapt",
        "candidate_representation": protocol.candidate_representation,
        "route_id": route_id,
        "actual_facade_execution_count": 4,
        "primary": validations["primary"],
        "fresh_leg": validations["fresh_leg"],
        "resumed": validations["resumed"],
        "g5_plateau_witness": validations["g5_plateau_witness"],
        "g5_first_interior_witness_receipt": witness_receipt,
        "bounded_fresh_comparison_sha256": fresh_comparison["sha256"],
        "authenticated_resume_comparison_sha256": (
            resume_comparison["sha256"]
        ),
        "g5_witness_prefix_comparison_sha256": witness_comparison["sha256"],
        "resumed_from_round": 2,
        "resumed_final_round": 3,
        "post_resume_controller_round_count": 1,
        "g5_witness_resumed_from_round": 3,
        "g5_witness_final_round": witness_round,
        "g5_witness_post_resume_controller_round_count": witness_round - 3,
        "short_prefix_aggregate_g5_claimed": False,
        "strict_aggregate_g5_witness_passed": True,
        "bounded_trajectory_identity_equal": True,
        "authenticated_resume_identity_equal": True,
        "g5_witness_prefix_identity_equal": True,
        "status": "passed",
    }


def _run_append_case(
    *,
    case_id: str,
    adapter: Any,
    problem: Any,
    work_root: Path,
) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt import (
        AppendAdaptRequest,
        run_append_adapt,
    )
    from pipelines.static_adapt.ra_adapt.replay_evidence import (
        compare_bounded_controller_replays,
    )
    from pipelines.static_adapt.sr_snake import (
        SRExecutionPolicy,
        SRStopPolicy,
    )

    case_root = work_root / case_id
    case_root.mkdir()
    results = {}
    for stem in ("primary", "independent_replay"):
        results[stem] = run_append_adapt(
            problem,
            AppendAdaptRequest(
                adapter=adapter,
                execution=SRExecutionPolicy(
                    stop=SRStopPolicy(maximum_controller_rounds=2)
                ),
                observation=_observation(case_root, stem=stem),
            ),
        )
    payloads = {
        stem: result.to_dict() for stem, result in results.items()
    }
    validations = {
        stem: _validate_observed_run(
            case_id=f"{case_id}__{stem}",
            route_id="append_macro",
            payload=payload,
            checkpoint_path=case_root / f"{stem}.checkpoint.json",
            ledger_path=case_root / f"{stem}.ledger.json",
            append=True,
            expected_rounds=2,
        )
        for stem, payload in payloads.items()
    }
    first_trajectory = _trajectory(payloads["primary"], append=True)
    second_trajectory = _trajectory(
        payloads["independent_replay"], append=True
    )
    comparison = compare_bounded_controller_replays(
        _validated_replay(
            payloads["primary"], append=True, expected_rounds=2
        ),
        _validated_replay(
            payloads["independent_replay"],
            append=True,
            expected_rounds=2,
        ),
        controller_round=2,
    )
    if first_trajectory != second_trajectory or (
        comparison.get("matched") is not True
    ):
        raise PreflightError(
            f"{case_id} independent reconstruction replay failed."
        )
    protocol = _mapping(
        payloads["primary"]["protocol"], label=f"{case_id} protocol"
    )
    return {
        "case_id": case_id,
        "method_family": "append_adapt",
        "candidate_representation": protocol[
            "candidate_representation"
        ],
        "route_id": "append_endpoint_only",
        "actual_facade_execution_count": 2,
        "primary": validations["primary"],
        "independent_replay": validations["independent_replay"],
        "bounded_replay_comparison_sha256": comparison["sha256"],
        "full_trajectory_identity_equal": True,
        "append_resume_boundary_status": (
            "authenticated_reconstruction_only_verified"
        ),
        "status": "passed",
    }


def _run_p3(*, v8_root: Path, work_root: Path) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt.adapters import (
        MacroCandidateAdapter,
        SinglePauliWordCandidateAdapter,
    )
    from pipelines.static_adapt.ra_adapt.bundles import (
        load_validated_bundle_protocol,
    )

    measured_protocol_root = (
        v8_root / "ra_repair_measured_late_v1" / "protocols"
    )
    macro_protocol = (
        measured_protocol_root
        / "validation__strong_weak_u8__nph3__ra_macro_plateau.json"
    )
    singleton_protocol = (
        measured_protocol_root
        / "validation__strong_weak_u8__nph3__singleton_plateau.json"
    )
    append_problem = _problem_from_protocol(
        load_validated_bundle_protocol(macro_protocol)
    )
    cases = [
        _run_ra_case(
            case_id="p3_ra_macro_plateau",
            route_id="ra_macro_plateau",
            protocol_path=macro_protocol,
            work_root=work_root,
        ),
        _run_ra_case(
            case_id="p3_ra_singleton_plateau",
            route_id="singleton_plateau",
            protocol_path=singleton_protocol,
            work_root=work_root,
        ),
        _run_append_case(
            case_id="p3_append_macro",
            adapter=MacroCandidateAdapter(),
            problem=append_problem,
            work_root=work_root,
        ),
        _run_append_case(
            case_id="p3_append_singleton",
            adapter=SinglePauliWordCandidateAdapter(),
            problem=append_problem,
            work_root=work_root,
        ),
    ]
    if (
        len(cases) != 4
        or any(case.get("status") != "passed" for case in cases)
        or sum(
            int(case["actual_facade_execution_count"])
            for case in cases
        )
        != 12
    ):
        raise PreflightError("P3 did not close all four actual smokes.")
    return digested(
        {
            "schema": P3_SCHEMA,
            "status": "passed",
            "nph": 3,
            "actual_facade_execution_count": 12,
            "case_count": 4,
            "cases": cases,
            "short_ra_prefix_role": (
                "chart_replay_and_g11_only_no_aggregate_g5_claim_v1"
            ),
            "g5_first_interior_bounds_by_representation": (
                G5_FIRST_INTERIOR_ROUND_BY_REPRESENTATION
            ),
            "strict_g5_first_interior_witness_count": 2,
            "exact_chart_verified": True,
            "estimator_ledger_closure_verified": True,
            "checkpoint_replay_boundaries_verified": True,
        }
    )


def build_preflight_receipt(
    *,
    repo_root: Path,
    v8_root: Path,
) -> dict[str, Any]:
    authority = validate_v8_authority(repo_root, v8_root=v8_root)
    control_plane = package_control_plane_receipt(Path(__file__).parent)
    immutable_before = _immutable_tree_snapshot(v8_root)
    with tempfile.TemporaryDirectory(
        prefix="paper_i_study1_scientific_preflight__"
    ) as raw:
        isolation_root = Path(raw)
        runtime_v8_root = isolation_root / "v8_runtime_copy"
        shutil.copytree(v8_root, runtime_v8_root, copy_function=shutil.copy2)
        work_root = isolation_root / "p3_observations"
        work_root.mkdir()
        execution_cwd = isolation_root / "execution_cwd"
        execution_cwd.mkdir()
        cache_root = isolation_root / "cache"
        environment = {
            "STATIC_ADAPT_HH_POOL_CACHE_DIR": str(
                cache_root / "hh_pool"
            ),
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "disk",
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR": str(
                cache_root / "candidate_records"
            ),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "MPLCONFIGDIR": str(cache_root / "matplotlib"),
            "NUMBA_CACHE_DIR": str(cache_root / "numba"),
        }
        with _temporary_environment(environment), _working_directory(
            execution_cwd
        ):
            p2 = _run_p2()
            p3 = _run_p3(
                v8_root=runtime_v8_root,
                work_root=work_root,
            )
    post_authority = validate_v8_authority(repo_root, v8_root=v8_root)
    immutable_after = _immutable_tree_snapshot(v8_root)
    if (
        immutable_after != immutable_before
        or post_authority["final_receipt_binding"]
        != authority["final_receipt_binding"]
        or post_authority["objective_gate_authority"]["sha256"]
        != authority["objective_gate_authority"]["sha256"]
    ):
        raise PreflightError(
            "P2/P3 changed immutable v8 authority content or stat identity."
        )
    return digested(
        {
            "schema": PREFLIGHT_SCHEMA,
            "package_id": PACKAGE_ID,
            "materialization_revision": (
                authority["final_receipt"]["materialization_revision"]
            ),
            "v8_final_receipt_canonical_sha256": authority[
                "final_receipt_binding"
            ]["canonical_sha256"],
            "v8_final_receipt_file_sha256": authority[
                "final_receipt_binding"
            ]["file_sha256"],
            "study1_objective_gate_authority_sha256": authority[
                "objective_gate_authority"
            ]["sha256"],
            "package_control_plane_sha256": control_plane["sha256"],
            "v8_immutable_tree_snapshot_sha256": canonical_sha256(
                immutable_before
            ),
            "v8_immutable_tree_entry_count": immutable_before[
                "entry_count"
            ],
            "v8_content_and_stat_identity_unchanged": True,
            "runtime_materialization_mode": (
                "temporary_byte_copy_with_isolated_caches_v1"
            ),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "p2": p2,
            "p3": p3,
            "p2_passed": True,
            "p3_passed": True,
            "all_preflight_smokes_passed": True,
        }
    )


def _parser() -> argparse.ArgumentParser:
    repo_root = repo_root_from_script(__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v8-root",
        type=Path,
        default=repo_root / V8_RELATIVE_ROOT,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = repo_root_from_script(__file__)
    v8_root = args.v8_root.resolve()
    receipt = build_preflight_receipt(
        repo_root=repo_root,
        v8_root=v8_root,
    )
    _write_exclusive_json(args.output.resolve(), receipt)
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(args.output.resolve()),
                "sha256": receipt["sha256"],
                "p2_sha256": receipt["p2"]["sha256"],
                "p3_sha256": receipt["p3"]["sha256"],
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

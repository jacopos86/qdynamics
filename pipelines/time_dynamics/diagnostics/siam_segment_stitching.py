"""Digest-bound continuation lineage and stitching for the SIAM Nb4 pilot.

The production pilot is deliberately resumable: the short ``t<=1`` proof leg
may use one activation cut, while a later ``1<=t<=5`` leg may continue the
same state and controller history at another predeclared cut.  This module
keeps that convenience from weakening the evidence boundary.  Every resumed
runner artifact identifies the exact prior bytes and resume payload, and the
offline stitcher refuses any physics, method, numerical, controller, pool, or
code drift.

Stitching is row based.  The first leg's terminal checkpoint deliberately has
no solve/decision/integration work; the resumed leg evaluates that shared
checkpoint.  Therefore the stitcher retains the *later* boundary row, rather
than summing segment summaries or retaining both copies.  This yields exactly
500 intervals on the canonical 501-point Nb4 grid and prevents boundary costs
from being omitted or counted twice.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.resume import (
    CONTROLLER_RESUME_SCHEMA_V1,
    RESUME_SCHEMA_V2,
)
from pipelines.time_dynamics.run_lock import (
    comparison_fingerprint,
    physics_fingerprint,
)
from pipelines.time_dynamics.paper_ii_runs import (
    SIAM_NB4_CONTINUATION_SEGMENT,
    SIAM_NB4_SCREEN_SEGMENT,
)
from pipelines.time_dynamics.siam_seed import load_siam_seed_manifest


SIAM_SEGMENT_LINEAGE_V1 = "siam_raw_segment_lineage_v1"
SIAM_NB4_STITCH_V1 = "siam_nb4_raw_segment_stitch_v1"
SIAM_NB4_STITCH_COMPLETION_V1 = "siam_nb4_stitched_completion_receipt_v1"
SIAM_NB4_T_FINAL = 5.0
SIAM_NB4_POINT_COUNT = 501
SIAM_NB4_TIMES = tuple(
    float(value)
    for value in np.linspace(0.0, SIAM_NB4_T_FINAL, SIAM_NB4_POINT_COUNT)
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _document_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time_grid_sha256(times: Sequence[float]) -> str:
    return _document_sha256([float(value) for value in times])


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _rows(run: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(run.get("plot_rows") or ())
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("SIAM segment must contain nonempty checkpoint rows.")
    times = tuple(float(row["time"]) for row in rows)
    if any(not math.isfinite(value) for value in times) or any(
        right <= left for left, right in zip(times, times[1:])
    ):
        raise ValueError("SIAM segment checkpoint times must be finite and increasing.")
    indices = tuple(int(row["index"]) for row in rows)
    if any(right != left + 1 for left, right in zip(indices, indices[1:])):
        raise ValueError("SIAM segment controller indices must increase by one.")
    return rows


def _state_payload(row: Mapping[str, Any]) -> dict[str, list[float]]:
    real = row.get("psi_real")
    imag = row.get("psi_imag")
    if not isinstance(real, Sequence) or isinstance(real, (str, bytes)):
        raise ValueError("SIAM segment boundary lacks psi_real.")
    if not isinstance(imag, Sequence) or isinstance(imag, (str, bytes)):
        raise ValueError("SIAM segment boundary lacks psi_imag.")
    real_values = [float(value) for value in real]
    imag_values = [float(value) for value in imag]
    if len(real_values) != len(imag_values) or not real_values:
        raise ValueError("SIAM segment boundary state shape is invalid.")
    if not all(math.isfinite(value) for value in (*real_values, *imag_values)):
        raise ValueError("SIAM segment boundary state must be finite.")
    return {"psi_real": real_values, "psi_imag": imag_values}


def _state_sha256(row: Mapping[str, Any]) -> str:
    return _document_sha256(_state_payload(row))


def _ansatz_state_sha256(value: Any) -> str:
    """Hash the structural state while ignoring path-only provenance.

    The resumed state's provenance intentionally differs from the prior
    terminal state's provenance because it records the continuation source.
    Every executable ansatz field must nevertheless be identical.
    """

    state = deepcopy(dict(_mapping(value, name="SIAM ansatz state")))
    state.pop("provenance", None)
    return _document_sha256(state)


def _resume(run: Mapping[str, Any]) -> Mapping[str, Any]:
    resume = _mapping(run.get("resume_state"), name="SIAM resume_state")
    if str(resume.get("schema")) != RESUME_SCHEMA_V2:
        raise ValueError(f"SIAM segment requires resume schema {RESUME_SCHEMA_V2!r}.")
    if str(resume.get("problem_family", "")).strip().lower() != "siam":
        raise ValueError("SIAM segment resume payload has the wrong family.")
    controller = _mapping(
        resume.get("controller_state"), name="SIAM controller resume state"
    )
    if str(controller.get("schema")) != CONTROLLER_RESUME_SCHEMA_V1:
        raise ValueError("SIAM segment lacks a valid controller resume state.")
    return resume


def _activation_cut(run: Mapping[str, Any]) -> float:
    summary = _mapping(run.get("summary"), name="SIAM summary")
    support = _mapping(
        summary.get("support_patch_config"), name="SIAM support-patch config"
    )
    value = support.get("insertion_l2_cut")
    if isinstance(value, bool):
        raise ValueError("SIAM activation cut must be a finite positive number.")
    cut = float(value)
    if not math.isfinite(cut) or cut <= 0.0:
        raise ValueError("SIAM activation cut must be a finite positive number.")
    avqds_cut = support.get("avqds_l2_cut")
    if avqds_cut is not None and not math.isclose(
        float(avqds_cut), cut, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("SIAM AVQDS and insertion activation cuts disagree.")
    return cut


def _stable_segment_contract(run: Mapping[str, Any]) -> dict[str, Any]:
    lock = _mapping(run.get("run_lock"), name="SIAM run lock")
    physics = _mapping(lock.get("physics"), name="SIAM run-lock physics")
    shared = _mapping(
        lock.get("shared_protocol"), name="SIAM shared protocol"
    )
    policy = _mapping(lock.get("policy"), name="SIAM structural policy")
    if str(physics.get("family_key", "")).strip().lower() != "siam":
        raise ValueError("SIAM segment run lock has the wrong problem family.")
    if physics.get("exact_reference_json") is not None or physics.get(
        "exact_reference_sha256"
    ) is not None:
        raise ValueError("SIAM raw segments must remain exact-reference blind.")
    data_flow = _mapping(
        run.get("decision_data_flow"), name="SIAM decision-data-flow contract"
    )
    if bool(data_flow.get("uses_exact_reference_for_decision", True)):
        raise ValueError("SIAM raw segment used an exact reference online.")
    seed_sha256 = str(physics.get("seed_sha256") or "")
    if len(seed_sha256) != 64:
        raise ValueError("SIAM segment run lock lacks its seed SHA-256.")
    code_revision = str(lock.get("code_revision") or "").strip()
    if not code_revision:
        raise ValueError("SIAM segment run lock lacks its code revision.")
    summary = _mapping(run.get("summary"), name="SIAM summary")
    support_config = deepcopy(
        dict(
            _mapping(
                summary.get("support_patch_config"),
                name="SIAM support-patch config",
            )
        )
    )
    # The activation cut is the only permitted segment-local calibration
    # variable.  Remove its duplicated representations before comparison.
    support_config.pop("insertion_l2_cut", None)
    support_config.pop("avqds_l2_cut", None)
    generalized_exchange = support_config.get("generalized_exchange")
    if isinstance(generalized_exchange, Mapping):
        generalized_exchange = deepcopy(dict(generalized_exchange))
        score = generalized_exchange.get("score")
        if isinstance(score, Mapping):
            score = deepcopy(dict(score))
            score.pop("l2_cut", None)
            generalized_exchange["score"] = score
        support_config["generalized_exchange"] = generalized_exchange
    return {
        "seed_sha256": seed_sha256,
        "family_key": "siam",
        "n_ph_max": physics.get("n_ph_max"),
        "drive_profile": physics.get("drive_profile"),
        "structural_policy": str(policy.get("structural_policy") or ""),
        "numerics": deepcopy(lock.get("numerics")),
        "time_step_controller": deepcopy(shared.get("time_step_controller")),
        "candidate_pool": deepcopy(shared.get("candidate_pool")),
        "controller_config": deepcopy(summary.get("controller_config")),
        "structural_configuration_without_activation_cut": support_config,
        "parameterization_mode": str(
            _mapping(run.get("trajectory"), name="SIAM trajectory")
            .get("metadata", {})
            .get("parameterization_mode", "")
        ),
        "code_revision": code_revision,
    }


def _exchange_route_identity(
    run: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Return ``(route bundle id, route contract digest)`` for one raw segment.

    Route identity is read from ``run_lock.policy.guards``.  It is deliberately
    kept out of :func:`_stable_segment_contract`: that contract is embedded and
    digested inside already-sealed segment-lineage documents, so extending it
    would move ``source_protocol_sha256`` for evidence that never changed.  The
    stitcher enforces route agreement directly instead.
    """

    lock = _mapping(run.get("run_lock"), name="SIAM run lock")
    policy = lock.get("policy")
    guards = policy.get("guards") if isinstance(policy, Mapping) else None
    if not isinstance(guards, Mapping):
        return None, None
    contract = guards.get("exchange_route_contract")
    bundle_id = (
        str(contract.get("bundle_id") or "") or None
        if isinstance(contract, Mapping)
        else None
    )
    digest = guards.get("exchange_route_contract_sha256")
    return bundle_id, (None if digest is None else str(digest))


def _validate_run_lock_grid(
    run: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    lock = _mapping(run.get("run_lock"), name="SIAM run lock")
    physics = _mapping(lock.get("physics"), name="SIAM run-lock physics")
    grid = _mapping(physics.get("time_grid"), name="SIAM run-lock time grid")
    times = tuple(float(row["time"]) for row in rows)
    if int(grid.get("point_count", -1)) != len(times):
        raise ValueError("SIAM segment run-lock point count mismatch.")
    if not math.isclose(
        float(grid.get("t_initial", float("nan"))),
        times[0],
        rel_tol=0.0,
        abs_tol=0.0,
    ) or not math.isclose(
        float(grid.get("t_final", float("nan"))),
        times[-1],
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("SIAM segment run-lock grid endpoints mismatch.")
    if grid.get("ordered_values_sha256") != _time_grid_sha256(times):
        raise ValueError("SIAM segment run-lock ordered-grid digest mismatch.")


def prepare_siam_resume_lineage(
    prior_run_path: str | Path,
    prior_run: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the exact prior raw bytes and terminal resume payload."""

    source_path = Path(prior_run_path)
    try:
        decoded = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("SIAM prior run must be a readable JSON artifact.") from exc
    if decoded != dict(prior_run):
        raise ValueError("Loaded SIAM prior run does not match its source bytes.")
    rows = _rows(prior_run)
    _validate_run_lock_grid(prior_run, rows)
    resume = _resume(prior_run)
    terminal_time = float(rows[-1]["time"])
    if not math.isclose(
        float(resume.get("time", float("nan"))),
        terminal_time,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("SIAM prior resume time does not equal its terminal row.")
    controller = _mapping(
        resume.get("controller_state"), name="SIAM controller resume state"
    )
    if int(controller.get("checkpoint_index", -1)) != int(rows[-1].get("index", -2)):
        raise ValueError(
            "SIAM prior controller checkpoint does not equal its terminal row index."
        )
    contract = _stable_segment_contract(prior_run)
    core = {
        "schema": SIAM_SEGMENT_LINEAGE_V1,
        "source_run_path": str(source_path.resolve()),
        "source_run_sha256": _file_sha256(source_path),
        "source_resume_payload_sha256": _document_sha256(resume),
        "source_controller_state_sha256": _document_sha256(controller),
        "source_terminal_state_sha256": _state_sha256(rows[-1]),
        "source_terminal_ansatz_sha256": _ansatz_state_sha256(
            prior_run.get("final_state")
        ),
        "source_terminal_time": terminal_time,
        "source_terminal_index": int(rows[-1]["index"]),
        "source_protocol": contract,
        "source_protocol_sha256": _document_sha256(contract),
        "source_activation_l2_cut": _activation_cut(prior_run),
    }
    return {**core, "prepared_lineage_sha256": _document_sha256(core)}


def assert_siam_resume_code_revision(
    prepared: Mapping[str, Any],
    code_revision: str | None,
) -> None:
    """Fail at start-up when a continuation would fail the lineage guard.

    ``finalize_siam_resume_lineage`` compares the whole segment contract after
    propagation, and that contract carries the code revision.  A continuation
    launched from a different revision than its source segment is therefore
    certain to be rejected after the full run (2026-09-07: cluster 9721121
    propagated 401 SIAM points for 21 hours and wrote nothing).  Check the one
    field that is already known before the first time step.
    """

    protocol = _mapping(prepared.get("source_protocol"), name="SIAM source protocol")
    source_revision = str(protocol.get("code_revision") or "").strip()
    current = str(code_revision or "").strip()
    if not current:
        raise ValueError(
            "SIAM continuation requires --code-revision so its segment contract "
            f"can be checked against the source segment ({source_revision})."
        )
    if current != source_revision:
        raise ValueError(
            "SIAM continuation code revision "
            f"{current} differs from the source segment's {source_revision}; "
            "the post-run lineage guard would reject this segment.  Re-run the "
            "source segment at the current revision or pin the continuation to "
            "the source revision."
        )


def finalize_siam_resume_lineage(
    prepared: Mapping[str, Any],
    current_run: Mapping[str, Any],
    *,
    controller_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the realized continuation boundary and seal its lineage."""

    prepared_payload = dict(prepared)
    prepared_digest = prepared_payload.pop("prepared_lineage_sha256", None)
    if prepared_digest != _document_sha256(prepared_payload):
        raise ValueError("SIAM prepared segment-lineage digest mismatch.")
    if prepared_payload.get("schema") != SIAM_SEGMENT_LINEAGE_V1:
        raise ValueError("Unsupported SIAM segment-lineage schema.")
    rows = _rows(current_run)
    _validate_run_lock_grid(current_run, rows)
    current_contract = _stable_segment_contract(current_run)
    if current_contract != prepared_payload.get("source_protocol"):
        raise ValueError(
            "SIAM continuation changed seed, family, structural method, numerics, "
            "time-step controller, pool, parameterization, or code revision."
        )
    boundary_time = float(rows[0]["time"])
    if not math.isclose(
        boundary_time,
        float(prepared_payload["source_terminal_time"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("SIAM continuation does not start at the prior terminal time.")
    current_state_sha256 = _state_sha256(rows[0])
    if current_state_sha256 != prepared_payload["source_terminal_state_sha256"]:
        raise ValueError("SIAM continuation boundary state differs from its source.")
    current_ansatz_sha256 = _ansatz_state_sha256(current_run.get("initial_state"))
    if current_ansatz_sha256 != prepared_payload["source_terminal_ansatz_sha256"]:
        raise ValueError("SIAM continuation ansatz differs from its source terminal state.")
    controller_input_digest = _document_sha256(
        _mapping(controller_input, name="SIAM current controller input")
    )
    if controller_input_digest != prepared_payload["source_controller_state_sha256"]:
        raise ValueError("SIAM continuation controller input differs from its source.")
    if int(rows[0].get("index", -1)) != int(
        prepared_payload["source_terminal_index"]
    ):
        raise ValueError("SIAM continuation controller index is not linked exactly.")
    lock = _mapping(current_run.get("run_lock"), name="SIAM run lock")
    grid = _mapping(
        _mapping(lock.get("physics"), name="SIAM run-lock physics").get(
            "time_grid"
        ),
        name="SIAM run-lock time grid",
    )
    if not math.isclose(
        float(grid.get("t_initial", float("nan"))),
        boundary_time,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("SIAM continuation run lock has the wrong initial time.")
    core = {
        **prepared_payload,
        "prepared_lineage_sha256": prepared_digest,
        "current_initial_time": boundary_time,
        "current_initial_index": int(rows[0]["index"]),
        "current_initial_state_sha256": current_state_sha256,
        "current_initial_ansatz_sha256": current_ansatz_sha256,
        "current_controller_input_sha256": controller_input_digest,
        "current_activation_l2_cut": _activation_cut(current_run),
        "activation_cut_changed": not math.isclose(
            float(prepared_payload["source_activation_l2_cut"]),
            _activation_cut(current_run),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "boundary_link_pass": True,
    }
    return {**core, "lineage_sha256": _document_sha256(core)}


def _verify_lineage_document(lineage: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(lineage)
    digest = payload.pop("lineage_sha256", None)
    if digest != _document_sha256(payload):
        raise ValueError("SIAM segment-lineage digest mismatch.")
    if payload.get("schema") != SIAM_SEGMENT_LINEAGE_V1:
        raise ValueError("Unsupported SIAM segment-lineage schema.")
    if payload.get("boundary_link_pass") is not True:
        raise ValueError("SIAM segment lineage lacks a passed boundary link.")
    return payload


def _canonical_slice_indices(times: Sequence[float]) -> tuple[int, int]:
    values = np.asarray([float(value) for value in times], dtype=float)
    canonical = np.asarray(SIAM_NB4_TIMES, dtype=float)
    matches = np.flatnonzero(canonical == values[0])
    if matches.size != 1:
        raise ValueError("SIAM Nb4 segment starts outside the canonical grid.")
    start = int(matches[0])
    stop = start + int(values.size)
    if stop > canonical.size or not np.array_equal(values, canonical[start:stop]):
        raise ValueError("SIAM Nb4 segment is not an exact canonical grid slice.")
    return start, stop - 1


def _row_costs(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    def interval_value(
        row: Mapping[str, Any], parameter_field: str, integration_field: str
    ) -> int:
        value = row.get(parameter_field)
        return int(value) if value is not None else int(row.get(integration_field) or 0)

    return {
        "accepted_internal_substep_count": sum(
            interval_value(
                row,
                "parameter_control_substep_count",
                "integration_local_substep_count",
            )
            for row in rows
        ),
        "rhs_evaluation_count": sum(
            interval_value(
                row,
                "parameter_control_rhs_evaluation_count",
                "integration_rhs_evaluation_count",
            )
            for row in rows
        ),
        "attempted_internal_step_count": sum(
            interval_value(
                row,
                "parameter_control_attempted_substep_count",
                "integration_attempted_substep_count",
            )
            for row in rows
        ),
        "rejected_internal_step_count": sum(
            interval_value(
                row,
                "parameter_control_rejected_substep_count",
                "integration_rejected_substep_count",
            )
            for row in rows
        ),
        "accepted_patch_count": sum(
            int(row.get("patch_commit_count") or 0) for row in rows
        ),
        "accepted_append_count": sum(
            int(row.get("patch_append_commit_count") or 0) for row in rows
        ),
        "accepted_insert_count": sum(
            int(row.get("patch_append_commit_count") or 0) for row in rows
        ),
        "accepted_appended_coordinate_count": sum(
            int(row.get("patch_append_inserted_coordinate_count") or 0)
            for row in rows
        ),
        "accepted_inserted_coordinate_count": sum(
            int(row.get("patch_append_inserted_coordinate_count") or 0)
            for row in rows
        ),
        "accepted_delete_count": sum(
            int(row.get("patch_delete_commit_count") or 0) for row in rows
        ),
        "accepted_exchange_count": sum(
            int(row.get("patch_exchange_commit_count") or 0) for row in rows
        ),
        "accepted_deleted_coordinate_count": sum(
            int(row.get("patch_delete_deleted_coordinate_count") or 0)
            for row in rows
        ),
        "accepted_exchange_appended_coordinate_count": sum(
            int(row.get("patch_exchange_inserted_coordinate_count") or 0)
            for row in rows
        ),
        "accepted_exchange_deleted_coordinate_count": sum(
            int(row.get("patch_exchange_deleted_coordinate_count") or 0)
            for row in rows
        ),
        "uncured_local_subdivision_count": sum(
            1
            for row in rows
            if "not_cured"
            in str(row.get("integration_local_subdivision_reason") or "")
        ),
        "local_subdivision_applied_count": sum(
            1
            for row in rows
            if bool(row.get("integration_local_subdivision_applied", False))
        ),
        "parameter_controlled_interval_count": sum(
            1
            for row in rows
            if row.get("parameter_control_substep_count") is not None
        ),
        "parameter_control_substep_count": sum(
            int(row.get("parameter_control_substep_count") or 0) for row in rows
        ),
        "solve_repair_applied_count": sum(
            1 for row in rows if bool(row.get("solve_repair_applied", False))
        ),
        "solve_repair_unsupported_count": sum(
            1 for row in rows if bool(row.get("solve_repair_unsupported", False))
        ),
    }


def _canonical_full_lock(run_lock: Mapping[str, Any]) -> dict[str, Any]:
    lock = deepcopy(dict(run_lock))
    physics = deepcopy(dict(_mapping(lock.get("physics"), name="SIAM physics")))
    physics["time_grid"] = {
        "t_initial": 0.0,
        "t_final": SIAM_NB4_T_FINAL,
        "point_count": SIAM_NB4_POINT_COUNT,
        "ordered_values_sha256": _time_grid_sha256(SIAM_NB4_TIMES),
    }
    physics_digest = physics_fingerprint(physics)
    shared = deepcopy(
        dict(_mapping(lock.get("shared_protocol"), name="SIAM shared protocol"))
    )
    shared["physics_fingerprint"] = physics_digest
    lock["physics"] = physics
    lock["physics_fingerprint"] = physics_digest
    lock["shared_protocol"] = shared
    lock["comparison_fingerprint"] = comparison_fingerprint(shared)
    return lock


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise FileExistsError(f"Refusing to replace immutable SIAM stitch: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(
                    f"Refusing to replace immutable SIAM stitch: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def stitch_siam_nb4_segments(
    segment_paths: Sequence[str | Path],
    *,
    seed_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify and join raw SIAM Nb4 segments on the canonical 0..5 grid."""

    paths = tuple(Path(path) for path in segment_paths)
    if len(paths) < 2:
        raise ValueError("SIAM Nb4 stitching requires at least two raw segments.")
    runs: list[dict[str, Any]] = []
    source_sha256: list[str] = []
    for path in paths:
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid SIAM raw segment: {path}") from exc
        if not isinstance(run, dict):
            raise ValueError(f"SIAM raw segment must be a JSON object: {path}")
        runs.append(run)
        source_sha256.append(_file_sha256(path))

    seed = Path(seed_path)
    seed_manifest = load_siam_seed_manifest(seed)
    if int(seed_manifest.physical["n_bath"]) != 4:
        raise ValueError("Canonical SIAM production stitching requires n_bath=4.")
    seed_sha256 = _file_sha256(seed)

    contracts = [_stable_segment_contract(run) for run in runs]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError(
            "SIAM segments differ in seed, family, structural method, numerics, "
            "time-step controller, pool, parameterization, or code revision."
        )
    if contracts[0]["seed_sha256"] != seed_sha256:
        raise ValueError("SIAM segments do not bind the supplied Nb4 seed bytes.")
    route_identities = [_exchange_route_identity(run) for run in runs]
    if any(identity != route_identities[0] for identity in route_identities[1:]):
        raise ValueError(
            "SIAM segments were produced on different exchange routes; a stitched "
            "horizon must stay on one route contract."
        )

    row_groups = [_rows(run) for run in runs]
    for run, rows in zip(runs, row_groups):
        _validate_run_lock_grid(run, rows)
    index_ranges = [
        _canonical_slice_indices([float(row["time"]) for row in rows])
        for rows in row_groups
    ]
    for rows, bounds in zip(row_groups, index_ranges):
        expected_indices = tuple(range(bounds[0], bounds[1] + 1))
        if tuple(int(row["index"]) for row in rows) != expected_indices:
            raise ValueError(
                "SIAM Nb4 segment controller indices do not match the canonical grid."
            )
    if index_ranges[0][0] != 0 or index_ranges[-1][1] != (
        SIAM_NB4_POINT_COUNT - 1
    ):
        raise ValueError("SIAM Nb4 segments must span the complete canonical grid.")

    source_rows: list[dict[str, Any]] = []
    for index, (path, run, rows, digest, bounds) in enumerate(
        zip(paths, runs, row_groups, source_sha256, index_ranges)
    ):
        resume = _resume(run)
        if not math.isclose(
            float(resume.get("time", float("nan"))),
            float(rows[-1]["time"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("SIAM segment resume time is not its terminal time.")
        terminal_controller = _mapping(
            resume.get("controller_state"), name="SIAM terminal controller state"
        )
        if int(terminal_controller.get("checkpoint_index", -1)) != int(
            rows[-1]["index"]
        ):
            raise ValueError(
                "SIAM segment terminal controller index differs from its last row."
            )
        if index:
            if bounds[0] != index_ranges[index - 1][1]:
                raise ValueError(
                    "Adjacent SIAM segments must overlap at exactly one boundary."
                )
            previous = runs[index - 1]
            previous_rows = row_groups[index - 1]
            lineage = _verify_lineage_document(
                _mapping(run.get("segment_lineage"), name="SIAM segment lineage")
            )
            if lineage["source_run_sha256"] != source_sha256[index - 1]:
                raise ValueError("SIAM lineage source-run digest mismatch.")
            if lineage["source_resume_payload_sha256"] != _document_sha256(
                _resume(previous)
            ):
                raise ValueError("SIAM lineage source-resume digest mismatch.")
            if not str(lineage.get("source_run_path") or "").strip():
                raise ValueError("SIAM lineage source-run path is missing.")
            previous_controller = _mapping(
                _resume(previous).get("controller_state"),
                name="SIAM source controller state",
            )
            controller_digest = _document_sha256(previous_controller)
            if lineage["source_controller_state_sha256"] != controller_digest or (
                lineage["current_controller_input_sha256"] != controller_digest
            ):
                raise ValueError("SIAM lineage controller linkage mismatch.")
            previous_state_digest = _state_sha256(previous_rows[-1])
            if lineage["source_terminal_state_sha256"] != previous_state_digest or (
                lineage["current_initial_state_sha256"] != _state_sha256(rows[0])
            ):
                raise ValueError("SIAM lineage boundary-state digest mismatch.")
            if previous_state_digest != _state_sha256(rows[0]):
                raise ValueError("SIAM segment boundary state is not exact.")
            previous_ansatz_digest = _ansatz_state_sha256(
                previous.get("final_state")
            )
            current_ansatz_digest = _ansatz_state_sha256(run.get("initial_state"))
            if lineage["source_terminal_ansatz_sha256"] != (
                previous_ansatz_digest
            ) or lineage["current_initial_ansatz_sha256"] != current_ansatz_digest:
                raise ValueError("SIAM lineage boundary-ansatz digest mismatch.")
            if previous_ansatz_digest != current_ansatz_digest:
                raise ValueError("SIAM segment boundary ansatz is not exact.")
            if int(lineage["source_terminal_index"]) != int(rows[0]["index"]):
                raise ValueError("SIAM segment boundary controller index differs.")
            if not math.isclose(
                float(lineage["source_terminal_time"]),
                float(previous_rows[-1]["time"]),
                rel_tol=0.0,
                abs_tol=0.0,
            ) or not math.isclose(
                float(lineage["current_initial_time"]),
                float(rows[0]["time"]),
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError("SIAM lineage boundary time differs.")
            if lineage["source_protocol"] != contracts[index - 1] or (
                lineage["source_protocol"] != contracts[index]
            ):
                raise ValueError("SIAM lineage protocol contract mismatch.")
            if lineage["source_protocol_sha256"] != _document_sha256(
                lineage["source_protocol"]
            ):
                raise ValueError("SIAM lineage protocol digest mismatch.")
            source_cut = _activation_cut(previous)
            current_cut = _activation_cut(run)
            if not math.isclose(
                float(lineage["source_activation_l2_cut"]),
                source_cut,
                rel_tol=0.0,
                abs_tol=0.0,
            ) or not math.isclose(
                float(lineage["current_activation_l2_cut"]),
                current_cut,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError("SIAM lineage activation-cut schedule mismatch.")
            if bool(lineage["activation_cut_changed"]) != (
                not math.isclose(source_cut, current_cut, rel_tol=0.0, abs_tol=0.0)
            ):
                raise ValueError("SIAM lineage activation-cut transition mismatch.")
        source_rows.append(
            {
                "ordinal": int(index),
                "run_path": str(path.resolve()),
                "run_sha256": digest,
                "resume_payload_sha256": _document_sha256(resume),
                "t_initial": float(rows[0]["time"]),
                "t_final": float(rows[-1]["time"]),
                "point_count": int(len(rows)),
                "canonical_start_index": int(bounds[0]),
                "canonical_stop_index": int(bounds[1]),
                "activation_l2_cut": _activation_cut(run),
                "code_revision": contracts[index]["code_revision"],
            }
        )

    # Retain the resumed leg's boundary row.  It owns the solve/decision and
    # integration interval that the earlier terminal row intentionally omitted.
    combined_rows = [deepcopy(dict(row)) for row in row_groups[0]]
    for rows in row_groups[1:]:
        combined_rows = combined_rows[:-1] + [deepcopy(dict(row)) for row in rows]
    combined_times = tuple(float(row["time"]) for row in combined_rows)
    if combined_times != SIAM_NB4_TIMES:
        raise ValueError("Stitched SIAM rows do not equal the canonical 0..5 grid.")

    trajectory_groups: list[tuple[Mapping[str, Any], ...]] = []
    for run, rows in zip(runs, row_groups):
        trajectory = _mapping(run.get("trajectory"), name="SIAM trajectory")
        points = tuple(trajectory.get("points") or ())
        if len(points) != len(rows) or any(
            float(point["time"]) != float(row["time"])
            for point, row in zip(points, rows)
        ):
            raise ValueError("SIAM trajectory points and plot rows disagree.")
        trajectory_groups.append(points)
    combined_points = [deepcopy(dict(point)) for point in trajectory_groups[0]]
    for points in trajectory_groups[1:]:
        combined_points = combined_points[:-1] + [
            deepcopy(dict(point)) for point in points
        ]

    costs = _row_costs(combined_rows)
    stitch_core = {
        "schema": SIAM_NB4_STITCH_V1,
        "seed_path": str(seed.resolve()),
        "seed_sha256": seed_sha256,
        "source_segments": source_rows,
        "protocol": contracts[0],
        "protocol_sha256": _document_sha256(contracts[0]),
        "canonical_grid": {
            "t_initial": 0.0,
            "t_final": SIAM_NB4_T_FINAL,
            "point_count": SIAM_NB4_POINT_COUNT,
            "ordered_values_sha256": _time_grid_sha256(SIAM_NB4_TIMES),
        },
        "activation_cut_schedule": [
            {
                "segment_ordinal": row["ordinal"],
                "t_initial": row["t_initial"],
                "t_final": row["t_final"],
                "activation_l2_cut": row["activation_l2_cut"],
            }
            for row in source_rows
        ],
        "boundary_policy": "retain_resumed_row_drop_prior_terminal_v1",
        "row_costs": costs,
    }
    stitch_receipt = {
        **stitch_core,
        "stitch_sha256": _document_sha256(stitch_core),
    }

    stitched = deepcopy(runs[-1])
    stitched["run_lock"] = _canonical_full_lock(runs[-1]["run_lock"])
    stitched["initial_state"] = deepcopy(runs[0].get("initial_state"))
    stitched["plot_rows"] = combined_rows
    trajectory = deepcopy(dict(stitched["trajectory"]))
    trajectory["points"] = combined_points
    trajectory_metadata = deepcopy(dict(trajectory.get("metadata") or {}))
    trajectory_metadata["segment_stitching"] = {
        "schema": SIAM_NB4_STITCH_V1,
        "stitch_sha256": stitch_receipt["stitch_sha256"],
        "source_run_sha256": source_sha256,
    }
    trajectory["metadata"] = trajectory_metadata
    stitched["trajectory"] = trajectory
    summary = deepcopy(dict(_mapping(stitched.get("summary"), name="SIAM summary")))
    first_summary = _mapping(runs[0].get("summary"), name="SIAM first summary")
    summary.update(costs)
    summary.update(
        {
            "point_count": SIAM_NB4_POINT_COUNT,
            "time_initial": 0.0,
            "time_final": SIAM_NB4_T_FINAL,
            "energy_initial": float(combined_rows[0]["energy_expectation"]),
            "energy_final": float(combined_rows[-1]["energy_expectation"]),
            "active_parameter_count_initial": int(
                first_summary["active_parameter_count_initial"]
            ),
            "runtime_parameter_count_initial": int(
                first_summary["runtime_parameter_count_initial"]
            ),
            "runtime_pauli_parameter_count_initial": int(
                first_summary["runtime_pauli_parameter_count_initial"]
            ),
            "logical_parameter_count_initial": int(
                first_summary["logical_parameter_count_initial"]
            ),
        }
    )
    stitched["summary"] = summary
    stitched["segment_stitching"] = stitch_receipt
    stitched["source_artifact_json"] = str(seed.resolve())
    stitched["completion_receipt"] = {
        "schema": SIAM_NB4_STITCH_V1,
        "status": "completed",
        "success": True,
        "stitch_sha256": stitch_receipt["stitch_sha256"],
    }
    destination = None if output_path is None else Path(output_path)
    if destination is not None:
        _write_immutable_json(destination, stitched)
    return stitched


def _canonical_grid_contract() -> dict[str, Any]:
    return {
        "t_initial": 0.0,
        "t_final": SIAM_NB4_T_FINAL,
        "point_count": SIAM_NB4_POINT_COUNT,
        "ordered_values_sha256": _time_grid_sha256(SIAM_NB4_TIMES),
    }


def _require_sha256(value: Any, *, name: str) -> str:
    token = str(value or "")
    if len(token) != 64 or any(
        character not in "0123456789abcdef" for character in token
    ):
        raise ValueError(f"{name} is invalid.")
    return token


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _canonical_proof_arm(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token == "avqds":
        return "avqds"
    # ``exchange_q_compiled`` is the compiled-Q route's arm id.  It is the same
    # structural proof arm as ``exchange`` -- the route it ran on is carried
    # separately, by ``route_bundle_id``, never by collapsing the arm family.
    if token in {"apm", "exchange", "full_apm", "exchange_q_compiled"}:
        return "exchange"
    raise ValueError(f"SIAM stitched completion arm is invalid: {value!r}.")


def _registered_segment_for_source(
    source: Mapping[str, Any],
) -> str:
    observed = (
        float(source.get("t_initial", float("nan"))),
        float(source.get("t_final", float("nan"))),
        int(source.get("point_count", -1)),
        int(source.get("canonical_start_index", -1)),
        int(source.get("canonical_stop_index", -1)),
    )
    registered = (
        (
            SIAM_NB4_SCREEN_SEGMENT,
            0,
            len(SIAM_NB4_SCREEN_SEGMENT.times) - 1,
        ),
        (
            SIAM_NB4_CONTINUATION_SEGMENT,
            len(SIAM_NB4_SCREEN_SEGMENT.times) - 1,
            SIAM_NB4_POINT_COUNT - 1,
        ),
    )
    for segment, start, stop in registered:
        expected = (
            float(segment.times[0]),
            float(segment.times[-1]),
            len(segment.times),
            start,
            stop,
        )
        if observed == expected:
            return segment.segment_id
    raise ValueError(
        "SIAM stitched completion contains an unregistered source segment."
    )


def _verified_embedded_stitch(value: Any) -> dict[str, Any]:
    stitch = deepcopy(dict(_mapping(value, name="SIAM embedded stitch")))
    core = dict(stitch)
    digest = core.pop("stitch_sha256", None)
    if digest != _document_sha256(core):
        raise ValueError("SIAM embedded stitch digest mismatch.")
    if core.get("schema") != SIAM_NB4_STITCH_V1:
        raise ValueError("Unsupported SIAM embedded stitch schema.")
    if core.get("canonical_grid") != _canonical_grid_contract():
        raise ValueError("SIAM embedded stitch canonical grid mismatch.")
    seed_sha256 = _require_sha256(
        core.get("seed_sha256"), name="SIAM embedded stitch seed SHA-256"
    )
    protocol = _mapping(core.get("protocol"), name="SIAM stitch protocol")
    if protocol.get("seed_sha256") != seed_sha256:
        raise ValueError("SIAM embedded stitch protocol seed mismatch.")
    if core.get("protocol_sha256") != _document_sha256(protocol):
        raise ValueError("SIAM embedded stitch protocol digest mismatch.")
    sources = core.get("source_segments")
    if not isinstance(sources, list) or len(sources) != 2 or any(
        not isinstance(source, Mapping) for source in sources
    ):
        raise ValueError(
            "Canonical SIAM stitched completion requires screen and "
            "continuation sources."
        )
    if [int(source.get("ordinal", -1)) for source in sources] != [0, 1]:
        raise ValueError("SIAM embedded stitch source ordinals are not canonical.")
    if [_registered_segment_for_source(source) for source in sources] != [
        SIAM_NB4_SCREEN_SEGMENT.segment_id,
        SIAM_NB4_CONTINUATION_SEGMENT.segment_id,
    ]:
        raise ValueError("SIAM embedded stitch source order is not canonical.")
    for source in sources:
        _require_sha256(
            source.get("run_sha256"), name="SIAM stitch source trajectory SHA-256"
        )
    return stitch


def _load_json_mapping(path: Path, *, name: str) -> tuple[dict[str, Any], bytes]:
    try:
        content = path.read_bytes()
        decoded = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be readable UTF-8 JSON: {path}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return decoded, content


def _validate_source_action_receipt(
    receipt: Mapping[str, Any],
    **expected: Any,
) -> dict[str, Any]:
    # The CHTC package imports this module through the trajectory runner, so
    # importing its public validator at module import time would form a cycle.
    # Delaying the import preserves the required single receipt authority.
    from pipelines.time_dynamics.siam_chtc_package import (
        validate_siam_nb4_completion_receipt,
    )

    return validate_siam_nb4_completion_receipt(receipt, **expected)


def build_siam_nb4_stitch_completion_receipt(
    stitched_trajectory_path: str | Path,
    *,
    source_action_receipt_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Build the final-T=5 proof from a stitch and positive worker receipts.

    Source trajectories are deliberately not reopened here.  Their exact
    digests were already established by the stitcher and must equal the
    trajectory digests independently certified by the source-locked v3 worker
    receipts.
    """

    trajectory_path = Path(stitched_trajectory_path)
    stitched, stitched_bytes = _load_json_mapping(
        trajectory_path, name="SIAM stitched trajectory"
    )
    stitch = _verified_embedded_stitch(stitched.get("segment_stitching"))
    source_stitches = tuple(stitch["source_segments"])
    source_path_tokens = tuple(
        str(source.get("run_path") or "").strip() for source in source_stitches
    )
    if any(not token for token in source_path_tokens):
        raise ValueError("SIAM stitch source trajectory path is missing.")
    source_paths = tuple(Path(token) for token in source_path_tokens)
    seed_path_token = str(stitch.get("seed_path") or "").strip()
    if not seed_path_token:
        raise ValueError("SIAM stitch seed path is missing.")
    seed_path = Path(seed_path_token)
    rederived = stitch_siam_nb4_segments(source_paths, seed_path=seed_path)
    stitched_document_sha256 = _document_sha256(stitched)
    rederived_document_sha256 = _document_sha256(rederived)
    if rederived_document_sha256 != stitched_document_sha256:
        raise ValueError(
            "SIAM stitched trajectory differs from the source-derived payload."
        )
    rows = _rows(stitched)
    if tuple(float(row["time"]) for row in rows) != SIAM_NB4_TIMES:
        raise ValueError("SIAM stitched trajectory does not use the canonical grid.")
    _validate_run_lock_grid(stitched, rows)
    physics = _mapping(
        _mapping(stitched.get("run_lock"), name="SIAM stitched run lock").get(
            "physics"
        ),
        name="SIAM stitched physics",
    )
    if physics.get("time_grid") != _canonical_grid_contract():
        raise ValueError("SIAM stitched run lock canonical grid mismatch.")
    trajectory = _mapping(
        stitched.get("trajectory"), name="SIAM stitched trajectory payload"
    )
    points = tuple(trajectory.get("points") or ())
    if len(points) != SIAM_NB4_POINT_COUNT or tuple(
        float(point["time"]) for point in points
    ) != SIAM_NB4_TIMES:
        raise ValueError("SIAM stitched trajectory points are not canonical.")
    metadata_stitch = _mapping(
        _mapping(trajectory.get("metadata"), name="SIAM trajectory metadata").get(
            "segment_stitching"
        ),
        name="SIAM trajectory stitch metadata",
    )
    if metadata_stitch.get("stitch_sha256") != stitch["stitch_sha256"]:
        raise ValueError("SIAM trajectory metadata stitch digest mismatch.")
    embedded_completion = _mapping(
        stitched.get("completion_receipt"), name="SIAM stitched completion marker"
    )
    if (
        embedded_completion.get("schema") != SIAM_NB4_STITCH_V1
        or embedded_completion.get("status") != "completed"
        or embedded_completion.get("success") is not True
        or embedded_completion.get("stitch_sha256") != stitch["stitch_sha256"]
    ):
        raise ValueError("SIAM stitched trajectory lacks its positive stitch marker.")

    protocol = _mapping(stitch.get("protocol"), name="SIAM stitch protocol")
    arm = _canonical_proof_arm(protocol.get("structural_policy"))
    route_bundle_id, route_contract_sha256 = _exchange_route_identity(stitched)
    seed_sha256 = _require_sha256(
        stitch.get("seed_sha256"), name="SIAM stitch seed SHA-256"
    )
    receipt_paths = tuple(Path(path) for path in source_action_receipt_paths)
    if len(receipt_paths) != len(source_stitches):
        raise ValueError(
            "SIAM stitched completion requires one source action receipt per segment."
        )

    source_proofs: list[dict[str, Any]] = []
    action_ids: set[str] = set()
    search_profile_ids: set[str] = set()
    for ordinal, (source, receipt_path) in enumerate(
        zip(source_stitches, receipt_paths)
    ):
        receipt, receipt_bytes = _load_json_mapping(
            receipt_path, name=f"SIAM source action receipt {ordinal}"
        )
        expected_segment = _registered_segment_for_source(source)
        try:
            verified = _validate_source_action_receipt(
                receipt,
                expected_source_revision=str(source.get("code_revision") or ""),
                expected_seed_sha256=seed_sha256,
                expected_trajectory_sha256=str(source.get("run_sha256") or ""),
                expected_arm=arm,
                expected_segment=expected_segment,
            )
        except (ValueError, SystemExit) as exc:
            raise ValueError(
                f"SIAM source action receipt {ordinal} is not a positive "
                f"v3 proof: {exc}"
            ) from exc
        if float(verified["activation_cut"]) != float(
            source.get("activation_l2_cut", float("nan"))
        ):
            raise ValueError(
                f"SIAM source action receipt {ordinal} activation-cut mismatch."
            )
        action_id = str(verified["action_id"])
        if action_id in action_ids:
            raise ValueError("SIAM stitched completion repeats a source action ID.")
        action_ids.add(action_id)
        # The continuation inherits the screen's ansatz, so a stitch across two
        # different declared search budgets is not one trajectory.
        search_profile_ids.add(str(verified["search_profile_id"]))
        if len(search_profile_ids) != 1:
            raise ValueError(
                "SIAM stitched completion pairs segments from different search "
                f"profiles: {sorted(search_profile_ids)}."
            )
        source_proofs.append(
            {
                "ordinal": ordinal,
                "segment": expected_segment,
                "trajectory_sha256": str(source["run_sha256"]),
                "trajectory_bytes": int(verified["trajectory_bytes"]),
                "t_initial": float(source["t_initial"]),
                "t_final": float(source["t_final"]),
                "point_count": int(source["point_count"]),
                "canonical_start_index": int(source["canonical_start_index"]),
                "canonical_stop_index": int(source["canonical_stop_index"]),
                "activation_l2_cut": float(source["activation_l2_cut"]),
                "source_revision": str(source["code_revision"]),
                "completion_receipt_path": str(receipt_path.resolve()),
                "completion_receipt_file_bytes": len(receipt_bytes),
                "completion_receipt_file_sha256": hashlib.sha256(
                    receipt_bytes
                ).hexdigest(),
                "completion_receipt_file_utf8": receipt_bytes.decode("utf-8"),
                "completion_receipt_document_sha256": _document_sha256(receipt),
                "completion_receipt": deepcopy(receipt),
            }
        )

    core = {
        "schema": SIAM_NB4_STITCH_COMPLETION_V1,
        "status": "COMPLETED",
        "problem_family": "siam",
        "arm": arm,
        "route_bundle_id": route_bundle_id,
        "route_contract_sha256": route_contract_sha256,
        "stitched_trajectory": {
            "path": str(trajectory_path.resolve()),
            "sha256": hashlib.sha256(stitched_bytes).hexdigest(),
            "bytes": len(stitched_bytes),
            "canonical_document_sha256": stitched_document_sha256,
            "source_rederived_document_sha256": rederived_document_sha256,
        },
        "seed": {
            "path": str(stitch.get("seed_path") or ""),
            "sha256": seed_sha256,
        },
        "canonical_grid": _canonical_grid_contract(),
        "stitch": deepcopy(stitch),
        "source_segments": source_proofs,
        "checks": {
            "stitched_trajectory_digest_bound": True,
            "stitched_payload_rederived_from_sources": True,
            "embedded_stitch_digest_verified": True,
            "canonical_grid_verified": True,
            "seed_binding_verified": True,
            "arm_binding_verified": True,
            "all_source_segment_digests_bound": True,
            "all_source_completion_receipts_positive_v3": True,
        },
    }
    document = {**core, "completion_sha256": _document_sha256(core)}
    return validate_siam_nb4_stitch_completion_receipt(document)


def validate_siam_nb4_stitch_completion_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_trajectory_sha256: str | None = None,
    expected_seed_sha256: str | None = None,
    expected_arm: str | None = None,
) -> dict[str, Any]:
    """Validate a self-contained final-T=5 proof without opening trajectories."""

    document = deepcopy(dict(receipt))
    core = deepcopy(document)
    digest = core.pop("completion_sha256", None)
    if digest != _document_sha256(core):
        raise ValueError("SIAM stitched completion-receipt digest mismatch.")
    if core.get("schema") != SIAM_NB4_STITCH_COMPLETION_V1:
        raise ValueError("Unsupported SIAM stitched completion-receipt schema.")
    if core.get("status") != "COMPLETED" or core.get("problem_family") != "siam":
        raise ValueError("SIAM stitched completion receipt is not positive.")
    arm = _canonical_proof_arm(core.get("arm"))
    if core.get("arm") != arm:
        raise ValueError("SIAM stitched completion arm is not canonical.")
    # Receipts sealed before the route guards existed simply omit these keys.
    route_bundle_id = core.get("route_bundle_id")
    if route_bundle_id is not None and not str(route_bundle_id).strip():
        raise ValueError("SIAM stitched completion route bundle id is empty.")
    route_contract_sha256 = core.get("route_contract_sha256")
    if route_contract_sha256 is not None:
        _require_sha256(
            route_contract_sha256, name="SIAM stitched completion route SHA-256"
        )

    trajectory = _mapping(
        core.get("stitched_trajectory"), name="SIAM stitched trajectory binding"
    )
    trajectory_sha256 = _require_sha256(
        trajectory.get("sha256"), name="SIAM stitched trajectory SHA-256"
    )
    _positive_integer(
        trajectory.get("bytes"), name="SIAM stitched trajectory byte count"
    )
    canonical_document_sha256 = _require_sha256(
        trajectory.get("canonical_document_sha256"),
        name="SIAM stitched canonical-document SHA-256",
    )
    source_rederived_document_sha256 = _require_sha256(
        trajectory.get("source_rederived_document_sha256"),
        name="SIAM source-rederived document SHA-256",
    )
    if canonical_document_sha256 != source_rederived_document_sha256:
        raise ValueError(
            "SIAM stitched and source-rederived document digests differ."
        )
    if not str(trajectory.get("path") or "").strip():
        raise ValueError("SIAM stitched trajectory path is missing.")
    seed = _mapping(core.get("seed"), name="SIAM stitched seed binding")
    seed_sha256 = _require_sha256(
        seed.get("sha256"), name="SIAM stitched seed SHA-256"
    )
    if not str(seed.get("path") or "").strip():
        raise ValueError("SIAM stitched seed path is missing.")
    if core.get("canonical_grid") != _canonical_grid_contract():
        raise ValueError("SIAM stitched completion canonical grid mismatch.")

    stitch = _verified_embedded_stitch(core.get("stitch"))
    if stitch.get("canonical_grid") != core.get("canonical_grid"):
        raise ValueError("SIAM completion and embedded stitch grids differ.")
    if stitch.get("seed_sha256") != seed_sha256 or stitch.get("seed_path") != seed.get(
        "path"
    ):
        raise ValueError("SIAM completion and embedded stitch seeds differ.")
    protocol = _mapping(stitch.get("protocol"), name="SIAM stitch protocol")
    if _canonical_proof_arm(protocol.get("structural_policy")) != arm:
        raise ValueError("SIAM completion and embedded stitch arms differ.")

    sources = core.get("source_segments")
    stitch_sources = stitch.get("source_segments")
    if not isinstance(sources, list) or not isinstance(stitch_sources, list) or len(
        sources
    ) != len(stitch_sources):
        raise ValueError("SIAM stitched completion source count mismatch.")
    action_ids: set[str] = set()
    search_profile_ids: set[str] = set()
    for ordinal, (source, stitch_source) in enumerate(zip(sources, stitch_sources)):
        source = _mapping(source, name=f"SIAM completion source {ordinal}")
        stitch_source = _mapping(
            stitch_source, name=f"SIAM embedded stitch source {ordinal}"
        )
        if int(source.get("ordinal", -1)) != ordinal:
            raise ValueError("SIAM completion source ordinal mismatch.")
        expected_segment = _registered_segment_for_source(stitch_source)
        if source.get("segment") != expected_segment:
            raise ValueError("SIAM completion source segment mismatch.")
        source_sha256 = _require_sha256(
            source.get("trajectory_sha256"),
            name="SIAM completion source trajectory SHA-256",
        )
        if source_sha256 != stitch_source.get("run_sha256"):
            raise ValueError("SIAM completion source trajectory SHA-256 mismatch.")
        _positive_integer(
            source.get("trajectory_bytes"),
            name="SIAM completion source trajectory byte count",
        )
        for field in (
            "t_initial",
            "t_final",
            "point_count",
            "canonical_start_index",
            "canonical_stop_index",
            "activation_l2_cut",
            "source_revision",
        ):
            stitch_field = "code_revision" if field == "source_revision" else field
            if source.get(field) != stitch_source.get(stitch_field):
                raise ValueError(f"SIAM completion source {field} mismatch.")
        if not str(source.get("completion_receipt_path") or "").strip():
            raise ValueError("SIAM source completion-receipt path is missing.")
        receipt_file_bytes = _positive_integer(
            source.get("completion_receipt_file_bytes"),
            name="SIAM source completion-receipt file byte count",
        )
        receipt_file_sha256 = _require_sha256(
            source.get("completion_receipt_file_sha256"),
            name="SIAM source completion-receipt file SHA-256",
        )
        receipt_file_utf8 = source.get("completion_receipt_file_utf8")
        if not isinstance(receipt_file_utf8, str):
            raise ValueError("SIAM source completion-receipt file bytes are missing.")
        encoded_receipt = receipt_file_utf8.encode("utf-8")
        if len(encoded_receipt) != receipt_file_bytes:
            raise ValueError("SIAM source completion-receipt file byte count mismatch.")
        if hashlib.sha256(encoded_receipt).hexdigest() != receipt_file_sha256:
            raise ValueError("SIAM source completion-receipt file SHA-256 mismatch.")
        try:
            receipt_from_file = json.loads(receipt_file_utf8)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "SIAM embedded source completion-receipt file is invalid JSON."
            ) from exc
        action_receipt = _mapping(
            source.get("completion_receipt"),
            name=f"SIAM embedded source action receipt {ordinal}",
        )
        if receipt_from_file != dict(action_receipt):
            raise ValueError(
                "SIAM source completion-receipt file and document differ."
            )
        if source.get("completion_receipt_document_sha256") != _document_sha256(
            action_receipt
        ):
            raise ValueError("SIAM source completion-receipt document digest mismatch.")
        try:
            verified = _validate_source_action_receipt(
                action_receipt,
                expected_source_revision=str(stitch_source.get("code_revision") or ""),
                expected_seed_sha256=seed_sha256,
                expected_trajectory_sha256=source_sha256,
                expected_arm=arm,
                expected_segment=expected_segment,
            )
        except (ValueError, SystemExit) as exc:
            raise ValueError(
                f"SIAM embedded source action receipt {ordinal} is invalid: {exc}"
            ) from exc
        if int(verified["trajectory_bytes"]) != int(source["trajectory_bytes"]):
            raise ValueError("SIAM source trajectory byte count mismatch.")
        if float(verified["activation_cut"]) != float(
            stitch_source.get("activation_l2_cut", float("nan"))
        ):
            raise ValueError("SIAM source action-receipt activation-cut mismatch.")
        action_id = str(verified["action_id"])
        if action_id in action_ids:
            raise ValueError("SIAM stitched completion repeats a source action ID.")
        action_ids.add(action_id)
        search_profile_ids.add(str(verified["search_profile_id"]))
        if len(search_profile_ids) != 1:
            raise ValueError(
                "SIAM stitched completion pairs segments from different search "
                f"profiles: {sorted(search_profile_ids)}."
            )

    required_checks = {
        "stitched_trajectory_digest_bound",
        "stitched_payload_rederived_from_sources",
        "embedded_stitch_digest_verified",
        "canonical_grid_verified",
        "seed_binding_verified",
        "arm_binding_verified",
        "all_source_segment_digests_bound",
        "all_source_completion_receipts_positive_v3",
    }
    checks = _mapping(core.get("checks"), name="SIAM stitched completion checks")
    if set(checks) != required_checks or any(
        checks[field] is not True for field in checks
    ):
        raise ValueError("SIAM stitched completion lacks affirmative proof checks.")

    expected_values = {
        "trajectory SHA-256": (trajectory_sha256, expected_trajectory_sha256),
        "seed SHA-256": (seed_sha256, expected_seed_sha256),
        "arm": (
            arm,
            None if expected_arm is None else _canonical_proof_arm(expected_arm),
        ),
    }
    for name, (observed, expected) in expected_values.items():
        if expected is not None and observed != expected:
            raise ValueError(f"SIAM stitched completion {name} mismatch.")
    return document


def write_siam_nb4_stitch_completion_receipt(
    path: str | Path, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and immutably write one final-T=5 stitch completion proof."""

    verified = validate_siam_nb4_stitch_completion_receipt(receipt)
    _write_immutable_json(Path(path), verified)
    return verified


__all__ = [
    "SIAM_NB4_POINT_COUNT",
    "SIAM_NB4_STITCH_COMPLETION_V1",
    "SIAM_NB4_STITCH_V1",
    "SIAM_NB4_TIMES",
    "SIAM_NB4_T_FINAL",
    "SIAM_SEGMENT_LINEAGE_V1",
    "build_siam_nb4_stitch_completion_receipt",
    "assert_siam_resume_code_revision",
    "finalize_siam_resume_lineage",
    "prepare_siam_resume_lineage",
    "stitch_siam_nb4_segments",
    "validate_siam_nb4_stitch_completion_receipt",
    "write_siam_nb4_stitch_completion_receipt",
]

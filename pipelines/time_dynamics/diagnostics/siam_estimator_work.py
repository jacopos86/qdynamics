#!/usr/bin/env python3
"""Exact logical estimator-work accounting for one SIAM trajectory.

The Hubbard--Holstein cells already report the complete logical geometry-entry
budget: one full McLachlan geometry at support size ``p`` costs

    g(p) = p(p+1)/2 + p + 2

symmetric metric entries, force entries, and two scalar observables.  The same
convention is applied here to a SIAM run, reported at ``p = N_theta``, the
logical shared coordinate count, which for ``logical_shared`` *is* the
McLachlan geometry dimension.  The terminal Pauli rotation count is carried
separately and never enters the geometry dimension.

These are logical scalar-estimator entries, not hardware shots.  A statevector
run keeps no shot ledger, so the physical shot count is reported as explicitly
unavailable rather than as zero.

Segment discipline: the offline stitcher retains the *later* boundary row and
merges ``trajectory.points``, so estimator work for a continued trajectory is
only defined on the ONE stitched payload.  A bare continuation segment is
refused here, because summing two segments would double-charge the shared
boundary checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from pipelines.time_dynamics.diagnostics.paper_ii_four_route_comparison import (
    complete_estimator_work,
)
from pipelines.time_dynamics.diagnostics.siam_segment_stitching import (
    SIAM_NB4_STITCH_V1,
)


SIAM_ESTIMATOR_WORK_V1 = "siam_logical_estimator_work_v1"

# ``complete_estimator_work`` discriminates on exactly one thing: whether the
# route is AVQDS.  Every generalized-exchange arm takes the APM branch, whose
# canonical route key in the four-route accountant is ``q_compiled``.  This key
# selects an accounting branch; it does not assert a route-bundle identity for
# the run.
_APM_ACCOUNTING_ROUTE = "q_compiled"
_ARM_ROUTES: Mapping[str, str] = {
    "avqds": "avqds",
    "exchange": _APM_ACCOUNTING_ROUTE,
}

PHYSICAL_SHOTS_STATUS = "unavailable_statevector_no_shot_ledger"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _document_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(dict(value))).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _load_trajectory_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    """Parse and hash one immutable byte snapshot of a trajectory file."""

    try:
        content = path.read_bytes()
        value = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"SIAM trajectory is not readable JSON: {path}") from exc
    payload = dict(_mapping(value, name="SIAM trajectory"))
    return payload, hashlib.sha256(content).hexdigest()


def _stitch_evidence(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the stitch marker when this payload is a stitched trajectory."""

    receipt = payload.get("segment_stitching")
    if receipt is None:
        return None
    receipt = _mapping(receipt, name="SIAM segment stitching receipt")
    if str(receipt.get("schema")) != SIAM_NB4_STITCH_V1:
        raise ValueError("Unsupported SIAM segment-stitching schema.")
    digest = str(receipt.get("stitch_sha256") or "").strip()
    if len(digest) != 64:
        raise ValueError("SIAM stitched trajectory lacks a stitch digest.")
    metadata = _mapping(
        (_mapping(payload.get("trajectory"), name="SIAM trajectory body")).get(
            "metadata"
        )
        or {},
        name="SIAM trajectory metadata",
    )
    marker = _mapping(
        metadata.get("segment_stitching") or {},
        name="SIAM trajectory stitch marker",
    )
    if str(marker.get("stitch_sha256") or "") != digest:
        raise ValueError("SIAM stitched trajectory metadata digest mismatch.")
    return {
        "schema": SIAM_NB4_STITCH_V1,
        "stitch_sha256": digest,
        "source_run_sha256": [
            str(value) for value in (marker.get("source_run_sha256") or ())
        ],
    }


def _assert_whole_trajectory(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse a bare continuation segment; accept a whole or stitched run.

    Estimator work charges one checkpoint base per trajectory point.  Two raw
    segments share their boundary checkpoint, so only a single-segment
    registered run or the digest-bound stitched payload is a valid accounting
    unit.
    """

    stitch = _stitch_evidence(payload)
    if stitch is not None:
        return {"segment_role": "stitched", "segment_stitching": stitch}

    if payload.get("segment_lineage") is not None:
        raise ValueError(
            "Refusing estimator work on a bare SIAM continuation segment: "
            "account on the stitched payload so the shared boundary "
            "checkpoint is charged once."
        )
    trajectory = _mapping(payload.get("trajectory"), name="SIAM trajectory body")
    metadata = _mapping(
        trajectory.get("metadata") or {}, name="SIAM trajectory metadata"
    )
    if metadata.get("resume_lineage") is not None:
        raise ValueError(
            "Refusing estimator work on a resumed SIAM segment: "
            "account on the stitched payload so the shared boundary "
            "checkpoint is charged once."
        )
    rows = tuple(payload.get("plot_rows") or ())
    if not rows:
        raise ValueError("SIAM trajectory carries no checkpoint rows.")
    first_time = float(_mapping(rows[0], name="SIAM first checkpoint row")["time"])
    if not math.isfinite(first_time) or first_time != 0.0:
        raise ValueError(
            "Refusing estimator work on a SIAM trajectory that does not begin "
            "at t=0 and carries no stitch evidence."
        )
    return {"segment_role": "single_segment", "segment_stitching": None}


def build_siam_estimator_work_record(
    trajectory_path: str | Path,
    *,
    seed_path: str | Path | None = None,
    arm: str,
) -> dict[str, Any]:
    """Return the digest-bound logical estimator-work record for one SIAM run.

    ``arm`` is the registered structural arm id (``exchange`` or ``avqds``).  It
    must agree with the run lock, and it selects the accounting branch of
    :func:`complete_estimator_work`.
    """

    arm_id = str(arm).strip().lower()
    route = _ARM_ROUTES.get(arm_id)
    if route is None:
        raise ValueError(
            f"unknown SIAM arm {arm!r}; known: {sorted(_ARM_ROUTES)}"
        )

    trajectory_source = Path(trajectory_path)
    payload, trajectory_sha256 = _load_trajectory_snapshot(trajectory_source)

    recorded_seed = str(payload.get("source_artifact_json") or "").strip()
    if seed_path is None and not recorded_seed:
        raise ValueError("SIAM trajectory does not identify its seed artifact.")
    seed_source = Path(seed_path if seed_path is not None else recorded_seed)
    seed_sha256 = _file_sha256(seed_source)

    run_lock = _mapping(payload.get("run_lock"), name="SIAM run lock")
    physics = _mapping(run_lock.get("physics"), name="SIAM run-lock physics")
    if str(physics.get("family_key", "")).strip().lower() != "siam":
        raise ValueError("SIAM estimator work requires family_key='siam'.")
    if str(physics.get("seed_sha256", "")) != seed_sha256:
        raise ValueError("SIAM estimator-work seed digest differs from its run lock.")
    policy = _mapping(run_lock.get("policy"), name="SIAM run-lock policy")
    locked_arm = str(policy.get("structural_policy", "")).strip().lower()
    if locked_arm != arm_id:
        raise ValueError(
            f"SIAM run lock declares structural policy {locked_arm!r}, not {arm_id!r}."
        )

    initial_state = _mapping(payload.get("initial_state"), name="SIAM initial state")
    final_state = _mapping(payload.get("final_state"), name="SIAM final state")
    for state, name in ((initial_state, "initial"), (final_state, "final")):
        if str(state.get("parameterization_mode")) != "logical_shared":
            raise ValueError(
                f"SIAM estimator work requires logical_shared {name} state."
            )

    segment = _assert_whole_trajectory(payload)

    summary = _mapping(payload.get("summary"), name="SIAM trajectory summary")
    n_theta_terminal = _integer(
        summary.get("logical_parameter_count_final"),
        name="SIAM summary logical_parameter_count_final",
    )
    terminal_pauli_rotation_count = _integer(
        summary.get("runtime_pauli_parameter_count_final"),
        name="SIAM summary runtime_pauli_parameter_count_final",
    )
    if n_theta_terminal < 0 or terminal_pauli_rotation_count < 0:
        raise ValueError("SIAM terminal coordinate counts cannot be negative.")
    if int(final_state.get("logical_parameter_count", -1)) != n_theta_terminal:
        raise ValueError("SIAM final-state logical count differs from the summary.")
    # In logical_shared mode the McLachlan geometry dimension is N_theta, not
    # the terminal Pauli rotation count.
    if int(final_state.get("runtime_parameter_count", -1)) != n_theta_terminal:
        raise ValueError(
            "SIAM logical_shared geometry dimension differs from N_theta."
        )

    estimator_work = dict(complete_estimator_work([payload], route=route))

    core = {
        "schema": SIAM_ESTIMATOR_WORK_V1,
        "trajectory_json": str(trajectory_source.resolve()),
        "trajectory_json_sha256": trajectory_sha256,
        "seed_json": str(seed_source.resolve()),
        "seed_json_sha256": seed_sha256,
        "problem_family": "siam",
        "parameterization_mode": "logical_shared",
        "arm": arm_id,
        "estimator_accounting_route": route,
        "segment_role": segment["segment_role"],
        "segment_stitching": segment["segment_stitching"],
        "n_theta_terminal": n_theta_terminal,
        "terminal_pauli_rotation_count": terminal_pauli_rotation_count,
        "geometry_entry_convention": "g(p) = p(p+1)/2 + p + 2",
        "estimator_work": estimator_work,
        "physical_shots": None,
        "physical_shots_status": PHYSICAL_SHOTS_STATUS,
    }
    return {**core, "estimator_work_sha256": _document_sha256(core)}


def verify_siam_estimator_work_record(payload: Mapping[str, Any]) -> bool:
    """Whether the embedded digest binds every other field of the document."""

    core = dict(payload)
    declared = str(core.pop("estimator_work_sha256", "") or "")
    return bool(declared) and declared == _document_sha256(core)


def write_siam_estimator_work_record(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write one digest-bound record idempotently without replacing evidence."""

    destination = Path(path)
    content = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == content:
            return destination
        raise FileExistsError(
            f"Refusing to replace immutable SIAM estimator-work record: {destination}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise FileExistsError(
                    "Refusing to replace immutable SIAM estimator-work record: "
                    f"{destination}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--seed-json", type=Path)
    parser.add_argument("--arm", required=True, choices=sorted(_ARM_ROUTES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_siam_estimator_work_record(
        args.trajectory_json,
        seed_path=args.seed_json,
        arm=args.arm,
    )
    write_siam_estimator_work_record(args.output, payload)
    work = payload["estimator_work"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "estimator_work_sha256": payload["estimator_work_sha256"],
                "n_theta_terminal": payload["n_theta_terminal"],
                "terminal_pauli_rotation_count": payload[
                    "terminal_pauli_rotation_count"
                ],
                "logical_total_estimator_work": work["logical_total_estimator_work"],
                "logical_total_estimator_work_status": work[
                    "logical_total_estimator_work_status"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PHYSICAL_SHOTS_STATUS",
    "SIAM_ESTIMATOR_WORK_V1",
    "build_siam_estimator_work_record",
    "verify_siam_estimator_work_record",
    "write_siam_estimator_work_record",
]

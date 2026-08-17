#!/usr/bin/env python3
"""Continue the Page-10 intermediate--strong route from round 50 to 70."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import traceback
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_macro_then_singleton_phase123_qiskit_phase23_"
    "no_lanes_tau1em4_r50_20260807_v1_chtc"
)
FETCH_ROOT = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_chtc_20260807_macro_then_singleton_phase123_qiskit_"
    "phase23_no_lanes_v1"
)
REGIME = os.environ.get("PAPER_I_PAGE10_CONTINUATION_REGIME", "intermediate_strong")
ATTEMPT = os.environ.get("PAPER_I_PAGE10_CONTINUATION_ATTEMPT", "v1")
REGIME_SOURCES = {
    "weak_strong": "9600705.3_weak_strong",
    "intermediate_strong": "9600705.4_intermediate_strong",
    "strong_strong_u8": "9600705.5_strong_strong_u8",
}
if REGIME not in REGIME_SOURCES:
    raise ValueError(f"Unsupported Page-10 strong-sector regime: {REGIME}")
EXECUTION_ID = (
    f"staged_phase23_qiskit_no_lanes__{REGIME}__nph7__"
    "ra_macro_then_singleton_phase123_qiskit_phase23_plateau"
)
JOB_PATH = PACKAGE_DIR / "jobs" / f"{EXECUTION_ID}.json"
ARCHIVE_PATH = FETCH_ROOT / f"{REGIME_SOURCES[REGIME]}.tar.gz"
RECEIPT_PATH = (
    FETCH_ROOT / REGIME_SOURCES[REGIME] / "worker_receipt.json"
)
OUTPUT_ROOT = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_page10_macro_then_singleton_phase23_qiskit_no_lanes_"
    f"{REGIME}_r50_to_r70_20260808_{ATTEMPT}"
)
SOURCE_ROUND = 50
TARGET_ROUND = 70
LOCAL_BUNDLE_ID = (
    "ra_adapt_macro_then_singleton_phase123_qiskit_phase23_no_lanes_"
    "tau1em4_r70_local_continuation_v1"
)

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"


class ContinuationError(RuntimeError):
    """Fail-closed local continuation error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("sha256", None)
    payload["sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContinuationError(f"Expected a JSON object: {path}")
    return value


def _verify_self_digest(value: Mapping[str, Any], *, label: str) -> None:
    expected = str(value.get("sha256", ""))
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    observed = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    if expected != observed:
        raise ContinuationError(f"{label} self-digest drifted.")


def _extract_resume() -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    receipt = _load_json(RECEIPT_PATH)
    _verify_self_digest(receipt, label="round-50 worker receipt")
    if (
        receipt.get("status") != "passed"
        or receipt.get("execution_id") != EXECUTION_ID
        or int(receipt.get("controller_rounds_completed", -1)) != SOURCE_ROUND
    ):
        raise ContinuationError("Round-50 worker receipt drifted.")
    rows = receipt.get("artifacts")
    if not isinstance(rows, list):
        raise ContinuationError("Round-50 artifact bindings are absent.")
    checkpoint_rows = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping)
        and "/checkpoints/current" in str(row.get("path", ""))
    ]
    if len(checkpoint_rows) != 3:
        raise ContinuationError(
            "Expected the checkpoint and exactly two resume sidecars."
        )
    by_archive_name = {f"./{row['path']}": row for row in checkpoint_rows}
    reuse_value = os.environ.get("PAPER_I_PAGE10_REUSE_RESUME_ROOT")
    resume_root = (
        Path(reuse_value).expanduser().resolve()
        if reuse_value
        else OUTPUT_ROOT / "resume_input"
    )
    if not reuse_value:
        resume_root.mkdir(parents=True, exist_ok=False)
    else:
        for row in checkpoint_rows:
            target = resume_root / Path(str(row["path"])).name
            if (
                not target.is_file()
                or target.is_symlink()
                or target.stat().st_size != int(row["size_bytes"])
                or _sha256_file(target) != row["sha256"]
            ):
                raise ContinuationError(
                    f"Reused resume artifact drifted: {target.name}"
                )
        checkpoint = resume_root / "current.json"
        checkpoint_row = next(
            row
            for row in checkpoint_rows
            if str(row["path"]).endswith("current.json")
        )
        print(f"reusing authenticated resume input at {resume_root}", flush=True)
        return checkpoint, checkpoint_row, checkpoint_rows
    observed: set[str] = set()
    with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
        for member in archive:
            row = by_archive_name.get(member.name)
            if row is None:
                continue
            if (
                member.name in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or int(member.size) != int(row["size_bytes"])
            ):
                raise ContinuationError(
                    f"Unsafe resume archive member: {member.name}"
                )
            source = archive.extractfile(member)
            if source is None:
                raise ContinuationError(
                    f"Unreadable resume archive member: {member.name}"
                )
            target = resume_root / Path(str(row["path"])).name
            digest = hashlib.sha256()
            size = 0
            print(f"extracting {target.name} ({member.size} bytes)", flush=True)
            with target.open("xb") as stream:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    stream.write(block)
                    digest.update(block)
                    size += len(block)
            if size != member.size or digest.hexdigest() != row["sha256"]:
                raise ContinuationError(
                    f"Extracted resume artifact drifted: {target.name}"
                )
            observed.add(member.name)
    if observed != set(by_archive_name):
        raise ContinuationError("Resume archive extraction is incomplete.")
    checkpoint = resume_root / "current.json"
    checkpoint_row = next(
        row for row in checkpoint_rows if str(row["path"]).endswith("current.json")
    )
    return checkpoint, checkpoint_row, checkpoint_rows


def _request_without_horizon(request: Any) -> dict[str, Any]:
    payload = request.to_dict()
    payload["execution"]["stop"].pop("maximum_controller_rounds", None)
    return payload


def run() -> None:
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    print("validating and extracting the authenticated round-50 resume", flush=True)
    checkpoint, checkpoint_binding, checkpoint_rows = _extract_resume()

    sys.path.insert(0, PACKAGE_DIR.as_posix())
    import run_cell as base  # noqa: PLC0415

    print("activating the sealed Page-10 source archive", flush=True)
    job, package_manifest, source_protocol, problem, temporary = base._prepare(
        JOB_PATH
    )
    try:
        from pipelines.static_adapt.ra_adapt import (  # noqa: PLC0415
            RAAdaptOperationalControls,
            run_ra_adapt,
        )
        from pipelines.static_adapt.ra_adapt.bundles import (  # noqa: PLC0415
            BundleCellSpec,
            _bundle_protocol_materialization_authority,
        )
        from pipelines.static_adapt.ra_adapt.contracts import (  # noqa: PLC0415
            _attach_validated_bundle_protocol_authority,
            canonical_sha256,
        )
        from pipelines.static_adapt.ra_adapt.engine import (  # noqa: PLC0415
            build_resolved_ra_protocol,
        )
        from pipelines.static_adapt.sr_snake import (  # noqa: PLC0415
            AcceptedStateResume,
            CheckpointObservation,
            EstimatorLedgerObservation,
            SRObservationPolicy,
        )

        from pipelines.static_adapt.sr_snake import (  # noqa: PLC0415
            _controller as controller_module,
        )

        original_match = controller_module._selection_state_matches_accepted

        def roundoff_compatible_match(selection: Any, accepted: Any) -> bool:
            tolerance = 128.0 * math.ulp(
                max(
                    1.0,
                    abs(selection.accepted_energy),
                    abs(accepted.accepted_energy),
                )
            )
            energy_compatible = math.isclose(
                selection.accepted_energy,
                accepted.accepted_energy,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            matched = bool(
                energy_compatible
                and original_match(
                    replace(
                        selection,
                        accepted_energy=accepted.accepted_energy,
                    ),
                    accepted,
                )
            )
            if (
                os.environ.get("PAPER_I_PAGE10_RESUME_DIAGNOSTIC") == "1"
                and not matched
            ):
                print(
                    "RESUME_STATE_DIFF remained after the roundoff-only repair",
                    flush=True,
                )
            return matched

        controller_module._selection_state_matches_accepted = (
            roundoff_compatible_match
        )

        source_receipt = source_protocol.bundle_materialization
        if source_receipt is None:
            raise ContinuationError("Source protocol has no bundle authority.")
        local_manifest = _digest(
            {
                "schema": "paper_i_page10_local_continuation_bundle_v1",
                "status": "passed_inert",
                "bundle_id": LOCAL_BUNDLE_ID,
                "source_package_id": package_manifest["package_id"],
                "source_package_manifest_sha256": package_manifest["sha256"],
                "source_protocol_sha256": source_protocol.sha256,
                "source_route_contract_sha256": source_protocol.route_contract[
                    "sha256"
                ],
                "source_round": SOURCE_ROUND,
                "target_round": TARGET_ROUND,
                "only_scientific_change": {
                    "path": "request.execution.stop.maximum_controller_rounds",
                    "before": SOURCE_ROUND,
                    "after": TARGET_ROUND,
                },
                "execution_authorized": True,
                "submission_authorized": False,
            }
        )
        request = replace(
            source_protocol.request,
            execution=replace(
                source_protocol.request.execution,
                stop=replace(
                    source_protocol.request.execution.stop,
                    maximum_controller_rounds=TARGET_ROUND,
                ),
            ),
        )
        cell = BundleCellSpec(
            cell_id=EXECUTION_ID,
            stage=f"page10_{REGIME}_r50_to_r70_local_continuation",
            regime_id=REGIME,
            nph=7,
            route_id=str(job["route_id"]),
            algorithm_id=str(job["algorithm_id"]),
            selector_family="ra_adapt",
            candidate_representation=str(job["candidate_representation"]),
            horizon=TARGET_ROUND,
            source_lock_id=source_receipt.source_lock_id,
        )
        authority = _bundle_protocol_materialization_authority(
            cell=cell,
            bundle_id=LOCAL_BUNDLE_ID,
            bundle_manifest_sha256=local_manifest["sha256"],
            source_locks_sha256=source_receipt.source_locks_sha256,
            source_lock_refs=source_protocol.source_locks,
            active_gradient_policy=source_protocol.active_gradient_policy,
            resource_weighting_scope=source_protocol.resource_weighting_scope,
        )
        protocol = build_resolved_ra_protocol(
            problem,
            request,
            materialization_authority=authority,
        )
        bound_authority = _bundle_protocol_materialization_authority(
            cell=cell,
            bundle_id=LOCAL_BUNDLE_ID,
            bundle_manifest_sha256=local_manifest["sha256"],
            source_locks_sha256=source_receipt.source_locks_sha256,
            source_lock_refs=source_protocol.source_locks,
            active_gradient_policy=source_protocol.active_gradient_policy,
            resource_weighting_scope=source_protocol.resource_weighting_scope,
            protocol_sha256=protocol.sha256,
        )
        protocol = _attach_validated_bundle_protocol_authority(
            protocol, bound_authority
        )
        if (
            protocol.horizon != TARGET_ROUND
            or protocol.route_contract["sha256"]
            != source_protocol.route_contract["sha256"]
            or _request_without_horizon(protocol.request)
            != _request_without_horizon(source_protocol.request)
            or int(source_protocol.horizon) != SOURCE_ROUND
        ):
            raise ContinuationError(
                "Continuation changed more than the authorized horizon."
            )
        _write_json(OUTPUT_ROOT / "bundle_manifest.json", local_manifest)
        _write_json(OUTPUT_ROOT / "resolved_protocol.json", protocol.to_dict())
        audit = _digest(
            {
                "schema": "paper_i_page10_r50_to_r70_source_lock_audit_v1",
                "status": "passed",
                "source_archive": {
                    "path": ARCHIVE_PATH.relative_to(REPO_ROOT).as_posix(),
                    "sha256": _sha256_file(ARCHIVE_PATH),
                    "size_bytes": ARCHIVE_PATH.stat().st_size,
                },
                "source_worker_receipt_sha256": _load_json(RECEIPT_PATH)[
                    "sha256"
                ],
                "source_checkpoint": checkpoint_binding,
                "source_checkpoint_sidecars": checkpoint_rows,
                "source_protocol_sha256": source_protocol.sha256,
                "target_protocol_sha256": protocol.sha256,
                "common_route_contract_sha256": protocol.route_contract[
                    "sha256"
                ],
                "source_round": SOURCE_ROUND,
                "target_round": TARGET_ROUND,
                "non_horizon_request_diff": [],
                "runtime_source_archive_sha256": package_manifest[
                    "source_archive"
                ]["sha256"],
                "resume_identity_guard_repair": {
                    "kind": "accepted_energy_roundoff_only_v1",
                    "absolute_tolerance": "128*ulp(max(1,abs(E1),abs(E2)))",
                    "all_non_energy_fields_exact": True,
                    "active_controller_source_sha256": _sha256_file(
                        REPO_ROOT
                        / "pipelines/static_adapt/sr_snake/_controller.py"
                    ),
                    "regression_test": (
                        "test/test_static_adapt_sr_snake_controller.py::"
                        "test_selection_state_accepts_only_roundoff_scale_"
                        "energy_replay"
                    ),
                },
                "execution_authorized_by": (
                    "explicit_user_request_2026-08-08"
                ),
                "submission_authorized": False,
            }
        )
        _write_json(OUTPUT_ROOT / "source_lock_audit.json", audit)

        checkpoint_out = OUTPUT_ROOT / "checkpoints/current.json"
        ledger_out = OUTPUT_ROOT / "result/estimator_ledger.json"
        checkpoint_out.parent.mkdir(parents=True, exist_ok=False)
        ledger_out.parent.mkdir(parents=True, exist_ok=False)
        controls = RAAdaptOperationalControls(
            maximum_controller_rounds=TARGET_ROUND,
            resume=AcceptedStateResume(
                checkpoint_path=checkpoint,
                checkpoint_sha256=str(checkpoint_binding["sha256"]),
            ),
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=checkpoint_out,
                    every_controller_rounds=1,
                    keep_history_tail=100,
                ),
                estimator_ledger=EstimatorLedgerObservation(path=ledger_out),
                resource_rounds=(TARGET_ROUND,),
            ),
        )
        print(
            "launching authenticated continuation at round 50; target round 70",
            flush=True,
        )
        source_root = Path(temporary.name) / "source"
        original = Path.cwd()
        os.chdir(source_root)
        try:
            result = run_ra_adapt(
                problem,
                protocol,
                operational_controls=controls,
            )
        finally:
            os.chdir(original)
        rounds = len(result.run.accepted_trajectory)
        if rounds != TARGET_ROUND:
            raise ContinuationError(
                f"Continuation stopped at round {rounds}, not {TARGET_ROUND}."
            )
        _write_json(OUTPUT_ROOT / "result/result.json", result.to_dict())
        if result.run.paper_i_summary is not None:
            _write_json(
                OUTPUT_ROOT / "summary/summary.json",
                result.run.paper_i_summary.to_dict(),
            )
        terminal = _digest(
            {
                "schema": "paper_i_page10_r50_to_r70_terminal_receipt_v1",
                "status": "passed",
                "source_round": SOURCE_ROUND,
                "target_round": TARGET_ROUND,
                "controller_rounds_completed": rounds,
                "source_checkpoint_sha256": checkpoint_binding["sha256"],
                "protocol_sha256": protocol.sha256,
                "route_contract_sha256": protocol.route_contract["sha256"],
                "final_energy": float(result.final_state.energy),
                "source_lock_audit_sha256": audit["sha256"],
            }
        )
        _write_json(OUTPUT_ROOT / "terminal_receipt.json", terminal)
        print(f"completed round {rounds}", flush=True)
    finally:
        temporary.cleanup()


def main() -> int:
    try:
        run()
        return 0
    except BaseException as exc:
        if OUTPUT_ROOT.is_dir():
            failure = _digest(
                {
                    "schema": "paper_i_page10_r50_to_r70_failure_v1",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            failure_path = OUTPUT_ROOT / "failure_receipt.json"
            if not failure_path.exists():
                _write_json(failure_path, failure)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

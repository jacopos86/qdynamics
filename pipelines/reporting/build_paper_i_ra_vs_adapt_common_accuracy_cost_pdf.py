#!/usr/bin/env python3
"""Build the Paper-I RA-vs-ADAPT common-attainable-error cost scorecard."""

from __future__ import annotations

import concurrent.futures
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from typing import Any, Mapping, Sequence

from pipelines.reporting import (
    build_paper_i_ra_adapt_stationary_core_master_pdf as master,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PROVENANCE = REPO_ROOT / (
    "output/pdf/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_"
    "evolving_partial_progress_provenance.json"
)
OUTPUT_DIR = REPO_ROOT / (
    "output/pdf/paper_i_ra_vs_adapt_common_accuracy_cost_20260729"
)
STEM = "paper_i_ra_vs_adapt_common_accuracy_cost_20260729"

REGIMES = (
    ("weak_weak", "Weak--weak", "WW"),
    ("intermediate_weak", "Intermediate--weak", "IW"),
    ("strong_weak_u8", "Strong--weak", "SW"),
    ("weak_strong", "Weak--strong", "WS"),
    ("intermediate_strong", "Intermediate--strong", "IS"),
    ("strong_strong_u8", "Strong--strong", "SS"),
)
REPRESENTATIONS = ("macro", "singleton")
POLICIES = ("no_insertion", "plateau")
QISKIT_FIELDS = ("N2q", "D2q", "Dc", "W1q")
COST_FIELDS = (*QISKIT_FIELDS, "S_alg")


class CommonAccuracyInputError(ValueError):
    """Raised when validated artifacts cannot support the scorecard."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CommonAccuracyInputError(
            f"{label} is unreadable: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CommonAccuracyInputError(f"{label} must be a JSON object")
    return payload


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CommonAccuracyInputError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise CommonAccuracyInputError(f"{label} must be an array")
    return value


def _integer(
    value: Any,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool):
        raise CommonAccuracyInputError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CommonAccuracyInputError(
            f"{label} must be an integer"
        ) from exc
    if result < minimum:
        raise CommonAccuracyInputError(
            f"{label} must be at least {minimum}"
        )
    return result


def _finite(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CommonAccuracyInputError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise CommonAccuracyInputError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise CommonAccuracyInputError(
            f"{label} must be at least {minimum}"
        )
    return result


def _representation(candidate_representation: str) -> str:
    if candidate_representation == "macro_generator_v1":
        return "macro"
    if candidate_representation == "single_pauli_word_v1":
        return "singleton"
    raise CommonAccuracyInputError(
        f"unknown candidate representation: {candidate_representation}"
    )


def _ra_policy(route_id: str) -> str | None:
    if route_id.endswith("_append_only"):
        return "no_insertion"
    if route_id.endswith("_plateau"):
        return "plateau"
    return None


def _safe_attempt(
    fetched_dir: Path,
    relative: str,
    *,
    execution_id: str,
) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or "." in pure.parts or ".." in pure.parts:
        raise CommonAccuracyInputError(
            f"{execution_id}: unsafe attempt path"
        )
    path = fetched_dir.joinpath(*pure.parts)
    if not path.is_file() or path.is_symlink():
        raise CommonAccuracyInputError(
            f"{execution_id}: attempt archive is unavailable"
        )
    return path


def _archive_json_members(
    attempt: Path,
    members: Mapping[str, str],
    *,
    execution_id: str,
) -> dict[str, tuple[dict[str, Any], str, int]]:
    """Read requested JSON members in one forward gzip pass.

    The attempt archives can be hundreds of MiB. Random seeks in ``r:gz``
    replay decompression for each requested member, so the report deliberately
    uses streaming mode and authenticates every requested payload during one
    pass.
    """

    wanted = {name: role for role, name in members.items()}
    loaded: dict[str, tuple[dict[str, Any], str, int]] = {}
    try:
        with tarfile.open(attempt, "r|gz") as archive:
            for member in archive:
                role = wanted.get(member.name)
                if role is None:
                    continue
                label = f"{execution_id} {role}"
                if role in loaded:
                    raise CommonAccuracyInputError(
                        f"{label} appears more than once in its archive"
                    )
                if not member.isfile():
                    raise CommonAccuracyInputError(
                        f"{label} is not a regular archive member"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise CommonAccuracyInputError(
                        f"{label} has no readable bytes"
                    )
                raw = stream.read()
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise CommonAccuracyInputError(
                        f"{label} is invalid JSON"
                    ) from exc
                if not isinstance(payload, dict):
                    raise CommonAccuracyInputError(
                        f"{label} must be a JSON object"
                    )
                loaded[role] = (payload, _sha256_bytes(raw), len(raw))
    except tarfile.TarError as exc:
        raise CommonAccuracyInputError(
            f"{execution_id}: attempt archive is unreadable"
        ) from exc
    missing = sorted(set(members) - set(loaded))
    if missing:
        raise CommonAccuracyInputError(
            f"{execution_id}: archive members are missing for "
            + ", ".join(missing)
        )
    return loaded


def _trace_from_summary(
    *,
    execution_id: str,
    method_family: str,
    summary: Mapping[str, Any],
    exact_energy: float,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    if method_family == "ra":
        if summary.get("schema") != "paper_i_run_summary_v1":
            raise CommonAccuracyInputError(
                f"{execution_id}: RA summary schema drifted"
            )
        rows = _sequence(
            summary.get("accepted_error_trace"),
            label=f"{execution_id} RA trace",
        )
        for expected_round, raw in enumerate(rows, start=1):
            row = _mapping(raw, label=f"{execution_id} RA trace row")
            controller_round = _integer(
                row.get("controller_round"),
                label=f"{execution_id} RA round",
                minimum=1,
            )
            energy = _finite(
                row.get("accepted_energy"),
                label=f"{execution_id} RA energy",
            )
            error = _finite(
                row.get("absolute_energy_error"),
                label=f"{execution_id} RA error",
                minimum=0.0,
            )
            if (
                controller_round != expected_round
                or not math.isclose(
                    error,
                    abs(energy - exact_energy),
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-12,
                )
            ):
                raise CommonAccuracyInputError(
                    f"{execution_id}: RA trace math drifted"
                )
            points.append(
                {
                    "controller_round": controller_round,
                    "absolute_energy_error": error,
                }
            )
    elif method_family == "append":
        if summary.get("schema") != "paper_i_append_run_summary_v1":
            raise CommonAccuracyInputError(
                f"{execution_id}: ADAPT summary schema drifted"
            )
        rows = _sequence(
            summary.get("accepted_history"),
            label=f"{execution_id} ADAPT history",
        )
        for expected_round, raw in enumerate(rows, start=1):
            row = _mapping(raw, label=f"{execution_id} ADAPT history row")
            controller_round = _integer(
                row.get("controller_round"),
                label=f"{execution_id} ADAPT round",
                minimum=1,
            )
            energy = _finite(
                row.get("energy_after"),
                label=f"{execution_id} ADAPT energy",
            )
            if controller_round != expected_round:
                raise CommonAccuracyInputError(
                    f"{execution_id}: ADAPT rounds drifted"
                )
            points.append(
                {
                    "controller_round": controller_round,
                    "absolute_energy_error": abs(energy - exact_energy),
                }
            )
    else:
        raise CommonAccuracyInputError(
            f"{execution_id}: unknown method family"
        )
    if len(points) != 50:
        raise CommonAccuracyInputError(
            f"{execution_id}: accepted trace is not 50 rounds"
        )
    return points


def _load_source_cell(
    source: Mapping[str, Any],
    records: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    execution_id = str(source.get("execution_id", ""))
    method_family = str(source.get("method_family", ""))
    route_id = str(source.get("route_id", ""))
    if method_family not in {"ra", "append"}:
        raise CommonAccuracyInputError(
            f"{execution_id}: unknown method family"
        )
    policy = "append" if method_family == "append" else _ra_policy(route_id)
    if policy is None:
        raise CommonAccuracyInputError(
            f"{execution_id}: noncomparison RA route reached the loader"
        )
    source_index = _integer(
        source.get("source_receipt_index"),
        label=f"{execution_id} source record index",
        minimum=1,
    )
    record = records.get(source_index)
    if record is None:
        raise CommonAccuracyInputError(
            f"{execution_id}: source record is unavailable"
        )
    package_dir = Path(str(record.get("package_dir", ""))).resolve()
    try:
        package_relative = package_dir.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise CommonAccuracyInputError(
            f"{execution_id}: package is outside the active repository"
        ) from exc
    attempt = _safe_attempt(
        Path(str(record.get("fetched_dir", ""))).resolve(),
        str(source.get("attempt_path", "")),
        execution_id=execution_id,
    )
    attempt_sha = _sha256_file(attempt)
    if attempt_sha != source.get("attempt_sha256"):
        raise CommonAccuracyInputError(
            f"{execution_id}: attempt archive hash drifted"
        )
    members = {
        "job": f"{package_relative}/jobs/{execution_id}.json",
        "worker": "worker_outputs/worker_receipt.json",
        "manifest": "worker_outputs/execution_manifest.json",
        "result": "worker_outputs/result.json",
        "summary": "worker_outputs/summary.json",
    }
    metadata_members = {
        role: name for role, name in members.items() if role != "result"
    }
    loaded = _archive_json_members(
        attempt,
        metadata_members,
        execution_id=execution_id,
    )
    expected_file_hashes = {
        "job": "job_file_sha256",
        "worker": "worker_receipt_file_sha256",
        "manifest": "execution_manifest_file_sha256",
        "summary": "summary_file_sha256",
    }
    for role, source_field in expected_file_hashes.items():
        if loaded[role][1] != source.get(source_field):
            raise CommonAccuracyInputError(
                f"{execution_id}: {role} file hash drifted"
            )
    job = loaded["job"][0]
    worker = loaded["worker"][0]
    summary = loaded["summary"][0]
    if (
        job.get("execution_id") != execution_id
        or job.get("package_id") != source.get("package_id")
        or worker.get("execution_id") != execution_id
        or worker.get("sha256") != source.get("worker_receipt_sha256")
        or str(job.get("route_id", "")) != route_id
        or str(job.get("regime_id", "")) != source.get("regime_id")
        or str(job.get("candidate_representation", ""))
        != source.get("candidate_representation")
    ):
        raise CommonAccuracyInputError(
            f"{execution_id}: archived identity drifted"
        )
    package_job = package_dir / "jobs" / f"{execution_id}.json"
    if (
        not package_job.is_file()
        or package_job.is_symlink()
        or _sha256_file(package_job) != loaded["job"][1]
    ):
        raise CommonAccuracyInputError(
            f"{execution_id}: package job binding drifted"
        )
    exact_energy = _finite(
        source.get("exact_same_cutoff_energy"),
        label=f"{execution_id} exact energy",
    )
    trace = _trace_from_summary(
        execution_id=execution_id,
        method_family=method_family,
        summary=summary,
        exact_energy=exact_energy,
    )
    terminal = _mapping(
        source.get("terminal"),
        label=f"{execution_id} terminal source row",
    )
    if (
        terminal.get("status") != "complete"
        or terminal.get("k") != 50
        or not math.isclose(
            _finite(
                terminal.get("error"),
                label=f"{execution_id} terminal error",
                minimum=0.0,
            ),
            float(trace[-1]["absolute_energy_error"]),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
    ):
        raise CommonAccuracyInputError(
            f"{execution_id}: terminal trace closure drifted"
        )
    print(f"validated source metadata: {execution_id}", flush=True)
    return {
        "execution_id": execution_id,
        "method_family": method_family,
        "policy": policy,
        "representation": _representation(
            str(source.get("candidate_representation", ""))
        ),
        "regime": str(source.get("regime_id", "")),
        "route_id": route_id,
        "package_id": str(source.get("package_id", "")),
        "package_dir": str(package_dir),
        "core_materialization_id": str(
            source.get("core_materialization_id", "")
        ),
        "exact_same_cutoff_energy": exact_energy,
        "trace": trace,
        "job": job,
        "summary": summary,
        "attempt_path": str(attempt),
        "result_member": members["result"],
        "result_file_sha256": str(source.get("result_file_sha256", "")),
        "source": {
            "source_receipt_index": source_index,
            "attempt_path": str(attempt),
            "attempt_sha256": attempt_sha,
            "job_file_sha256": loaded["job"][1],
            "result_file_sha256": str(source.get("result_file_sha256", "")),
            "summary_file_sha256": loaded["summary"][1],
            "worker_receipt_sha256": worker["sha256"],
        },
    }


def _load_cells() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provenance = _load_object(
        SOURCE_PROVENANCE,
        label="evolving stationary-core provenance",
    )
    if (
        provenance.get("schema")
        != (
            "paper_i_ra_adapt_stationary_core_master_"
            "cross_revision_partial_progress_v1"
        )
        or provenance.get("metric") != "same_cutoff_absolute_energy_error"
        or provenance.get("paper_evidence_adopted") is not False
    ):
        raise CommonAccuracyInputError(
            "evolving stationary-core provenance identity drifted"
        )
    records = {
        _integer(
            _mapping(raw, label="source record").get(
                "source_receipt_index"
            ),
            label="source record index",
            minimum=1,
        ): _mapping(raw, label="source record")
        for raw in _sequence(
            provenance.get("source_records"),
            label="source records",
        )
    }
    selected_sources: list[Mapping[str, Any]] = []
    for raw in _sequence(
        provenance.get("included_sources"),
        label="included sources",
    ):
        row = _mapping(raw, label="included source")
        family = str(row.get("method_family", ""))
        if family == "append" or (
            family == "ra"
            and _ra_policy(str(row.get("route_id", ""))) is not None
        ):
            selected_sources.append(row)
    cells: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_load_source_cell, source, records)
            for source in selected_sources
        ]
        for future in concurrent.futures.as_completed(futures):
            cells.append(future.result())
    cells.sort(
        key=lambda row: (
            REPRESENTATIONS.index(str(row["representation"])),
            tuple(regime[0] for regime in REGIMES).index(str(row["regime"])),
            str(row["policy"]),
        )
    )
    expected_append = {
        (representation, regime)
        for representation in REPRESENTATIONS
        for regime, _title, _abbreviation in REGIMES
    }
    observed_append = {
        (str(row["representation"]), str(row["regime"]))
        for row in cells
        if row["method_family"] == "append"
    }
    expected_none = {
        (representation, regime)
        for representation in REPRESENTATIONS
        for regime, _title, _abbreviation in REGIMES
    }
    observed_none = {
        (str(row["representation"]), str(row["regime"]))
        for row in cells
        if row["policy"] == "no_insertion"
    }
    if observed_append != expected_append or observed_none != expected_none:
        raise CommonAccuracyInputError(
            "ADAPT or RA no-insertion comparison matrix is incomplete"
        )
    return cells, {
        "path": str(SOURCE_PROVENANCE),
        "sha256": _sha256_file(SOURCE_PROVENANCE),
        "schema": provenance["schema"],
        "package_ids": provenance["package_ids"],
        "included_count": provenance["included_count"],
        "pending_count": provenance["pending_count"],
        "parameter_manifest": provenance["parameter_manifest"],
        "terminal_cost_policy": provenance["terminal_cost_policy"],
    }


def _capture_result_objects(
    cell: Mapping[str, Any],
    *,
    controller_round: int,
) -> dict[str, Any]:
    """Stream only the authenticated objects needed for one prefix."""

    import ijson
    from ijson.common import ObjectBuilder

    method_family = str(cell["method_family"])
    selected_index = controller_round - 1
    if method_family == "ra":
        indexed_targets = {
            "run.accepted_trajectory.item": selected_index,
            "run.scientific_replay.item": selected_index,
            "run.canonical_reporting.accepted_prefix_work.item": (
                selected_index
            ),
        }
        singleton_targets = {
            "run.canonical_reporting.reference_state",
            "run.route",
            "run.problem",
        }
    elif method_family == "append":
        indexed_targets = {
            (
                "result_payload.controller_replay_evidence."
                "signed_controller_round_prefixes.item"
            ): selected_index,
        }
        singleton_targets = set()
    else:
        raise CommonAccuracyInputError("unknown result projection family")

    expected = set(indexed_targets) | singleton_targets
    captured: dict[str, Any] = {}
    item_counts: defaultdict[str, int] = defaultdict(int)
    attempt = Path(str(cell["attempt_path"]))
    result_member = str(cell["result_member"])
    result_seen = False
    try:
        with tarfile.open(attempt, "r|gz") as archive:
            for member in archive:
                if member.name != result_member:
                    continue
                if result_seen:
                    raise CommonAccuracyInputError(
                        f"{cell['execution_id']}: duplicate result member"
                    )
                result_seen = True
                if not member.isfile():
                    raise CommonAccuracyInputError(
                        f"{cell['execution_id']}: result is not regular"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise CommonAccuracyInputError(
                        f"{cell['execution_id']}: result has no bytes"
                    )
                active_key: str | None = None
                builder: ObjectBuilder | None = None
                depth = 0
                for prefix, event, value in ijson.parse(
                    stream,
                    use_float=True,
                ):
                    if active_key is not None:
                        assert builder is not None
                        builder.event(event, value)
                        if event in {"start_map", "start_array"}:
                            depth += 1
                        elif event in {"end_map", "end_array"}:
                            depth -= 1
                            if depth == 0:
                                captured[active_key] = builder.value
                                active_key = None
                                builder = None
                                if set(captured) == expected:
                                    break
                        continue
                    should_capture = False
                    if prefix in singleton_targets and event == "start_map":
                        should_capture = True
                    elif (
                        prefix in indexed_targets
                        and event == "start_map"
                    ):
                        current_index = item_counts[prefix]
                        item_counts[prefix] += 1
                        should_capture = (
                            current_index == indexed_targets[prefix]
                        )
                    if should_capture:
                        active_key = prefix
                        builder = ObjectBuilder()
                        builder.event(event, value)
                        depth = 1
                break
    except (OSError, tarfile.TarError, ijson.JSONError) as exc:
        raise CommonAccuracyInputError(
            f"{cell['execution_id']}: streamed result projection failed"
        ) from exc
    if not result_seen or set(captured) != expected:
        missing = sorted(expected - set(captured))
        raise CommonAccuracyInputError(
            f"{cell['execution_id']}: streamed result projection is "
            f"incomplete ({', '.join(missing)})"
        )
    return captured


def _ra_prefix_from_archive(
    cell: Mapping[str, Any],
    *,
    controller_round: int,
) -> Any:
    projected = _capture_result_objects(
        cell,
        controller_round=controller_round,
    )
    selected_index = controller_round - 1
    trajectory = [{} for _ in range(50)]
    replay = [{} for _ in range(50)]
    work = [{} for _ in range(50)]
    trajectory[selected_index] = projected["run.accepted_trajectory.item"]
    replay[selected_index] = projected["run.scientific_replay.item"]
    work[selected_index] = projected[
        "run.canonical_reporting.accepted_prefix_work.item"
    ]
    result = {
        "run": {
            "accepted_trajectory": trajectory,
            "scientific_replay": replay,
            "canonical_reporting": {
                "accepted_prefix_work": work,
                "reference_state": projected[
                    "run.canonical_reporting.reference_state"
                ],
            },
            "route": projected["run.route"],
            "problem": projected["run.problem"],
        }
    }
    return master._ra_prefix(
        result,
        controller_round=controller_round,
    )


def _append_prefix_from_archive(
    cell: Mapping[str, Any],
    *,
    controller_round: int,
) -> Any:
    """Reconstruct one signed Append prefix without materializing its run."""

    import numpy as np

    from pipelines.reporting.paper_i_run_summary import (
        PaperIAlgorithmicWork,
        PaperIPrefixCompileInput,
        PaperIPrefixOperator,
        PaperIPrefixPauliTerm,
        PaperIReferenceState,
        PaperIWorkComponents,
    )
    from pipelines.static_adapt.estimator_call_ledger import (
        projective_state_fingerprint,
    )
    from pipelines.static_adapt.ra_adapt.append import (
        _validate_resolved_append_protocol,
    )
    from pipelines.static_adapt.ra_adapt.replay_evidence import (
        _prefix_replay_identity,
        _verify_signed,
    )

    projected = _capture_result_objects(
        cell,
        controller_round=controller_round,
    )
    selected = _mapping(
        projected[
            (
                "result_payload.controller_replay_evidence."
                "signed_controller_round_prefixes.item"
            )
        ],
        label="Append selected signed prefix",
    )
    expected_protocol = master._protocol_for_job(
        _mapping(cell["job"], label="Append job")
    )
    protocol = master._append_protocol_for_reporting(
        job=_mapping(cell["job"], label="Append job"),
        expected_protocol=expected_protocol,
    )
    problem = master._append_problem_from_protocol(protocol)
    (
        _request,
        _parent_inventory,
        executable_inventory,
        _lineage,
    ) = _validate_resolved_append_protocol(problem, protocol)

    verified_wrapper = _verify_signed(
        selected,
        name="selected Append controller prefix",
    )
    checkpoint = _verify_signed(
        verified_wrapper.get("active_prefix_checkpoint"),
        name="selected Append active-prefix checkpoint",
        signature_field="checkpoint_sha256",
    )
    route_identity = _mapping(
        verified_wrapper.get("route_identity"),
        label="Append selected route identity",
    )
    if (
        verified_wrapper.get("schema")
        != "paper_i_signed_controller_round_prefix_v1"
        or verified_wrapper.get("method_family") != "append_adapt"
        or verified_wrapper.get("controller_round") != controller_round
        or verified_wrapper.get("protocol_sha256") != protocol.sha256
        or verified_wrapper.get("problem_request_sha256")
        != protocol.problem.problem_request_sha256
        or verified_wrapper.get("source_checkpoint_sha256")
        != checkpoint.get("checkpoint_sha256")
        or route_identity
        != {
            "selector_identity": protocol.selector_identity,
            "selector_scope": protocol.selector_scope,
        }
        or checkpoint.get("schema")
        != "paper_i_signed_append_active_prefix_checkpoint_v1"
        or checkpoint.get("controller_round") != controller_round
        or checkpoint.get("protocol_sha256") != protocol.sha256
        or checkpoint.get("problem_request_sha256")
        != protocol.problem.problem_request_sha256
        or checkpoint.get("selector_identity")
        != protocol.selector_identity
        or checkpoint.get("selector_scope") != protocol.selector_scope
    ):
        raise CommonAccuracyInputError(
            f"{cell['execution_id']}: selected Append signature drifted"
        )
    replay_identity = _prefix_replay_identity(
        method_family="append_adapt",
        problem_request_sha256=protocol.problem.problem_request_sha256,
        route_identity=route_identity,
        controller_round=controller_round,
        operator_labels=checkpoint["accepted_operator_labels"],
        logical_parameters=checkpoint["logical_parameters"],
        runtime_parameters=checkpoint["runtime_parameters"],
        state_fingerprint=checkpoint["projective_state_fingerprint"],
        accepted_energy=float(checkpoint["accepted_energy"]),
    )
    if (
        replay_identity
        != verified_wrapper.get("prefix_replay_identity_sha256")
    ):
        raise CommonAccuracyInputError(
            f"{cell['execution_id']}: Append replay identity drifted"
        )

    labels = tuple(str(value) for value in checkpoint["accepted_operator_labels"])
    identities = tuple(
        str(value) for value in checkpoint["accepted_generator_identities"]
    )
    logical_parameters = tuple(
        _finite(value, label="Append selected logical parameter")
        for value in checkpoint["logical_parameters"]
    )
    runtime_parameters = tuple(
        _finite(value, label="Append selected runtime parameter")
        for value in checkpoint["runtime_parameters"]
    )
    if (
        len(labels) != controller_round
        or len(identities) != controller_round
        or len(logical_parameters) != controller_round
    ):
        raise CommonAccuracyInputError(
            f"{cell['execution_id']}: Append selected lineage drifted"
        )

    candidates = {
        str(candidate.label): candidate
        for candidate in executable_inventory.candidates
    }
    if len(candidates) != len(executable_inventory.candidates):
        raise CommonAccuracyInputError("Append executable pool duplicates labels")
    operators: list[Any] = []
    runtime_start = 0
    for logical_index, (label, identity) in enumerate(
        zip(labels, identities, strict=True)
    ):
        candidate = candidates.get(label)
        if (
            candidate is None
            or str(candidate.generator_identity) != identity
        ):
            raise CommonAccuracyInputError(
                f"{cell['execution_id']}: selected generator left the pool"
            )
        terms = tuple(
            PaperIPrefixPauliTerm(
                pauli_exyz=str(term["pauli_exyz"]),
                coefficient_real=_finite(
                    term.get("coeff_re"),
                    label="Append Pauli coefficient real",
                ),
                coefficient_imaginary=_finite(
                    term.get("coeff_im"),
                    label="Append Pauli coefficient imaginary",
                ),
                qubit_count=_integer(
                    term.get("nq"),
                    label="Append Pauli term qubit count",
                    minimum=1,
                ),
            )
            for term in candidate.serialized_terms_exyz
        )
        if not terms:
            raise CommonAccuracyInputError("Append candidate has no terms")
        operators.append(
            PaperIPrefixOperator(
                candidate_label=label,
                logical_index=logical_index,
                runtime_start=runtime_start,
                runtime_count=len(terms),
                execution_mode=str(candidate.execution_mode),
                runtime_terms=terms,
            )
        )
        runtime_start += len(terms)
    if runtime_start != len(runtime_parameters):
        raise CommonAccuracyInputError(
            f"{cell['execution_id']}: runtime parameter partition drifted"
        )

    reference_array = np.asarray(
        problem.reference_state.build_state(),
        dtype=complex,
    ).reshape(-1)
    reference_norm = float(np.linalg.norm(reference_array))
    if not math.isclose(
        reference_norm,
        1.0,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise CommonAccuracyInputError("Append reference is not normalized")
    reference_array = reference_array / reference_norm
    reference = PaperIReferenceState(
        amplitudes_real=tuple(float(value.real) for value in reference_array),
        amplitudes_imaginary=tuple(float(value.imag) for value in reference_array),
        qubit_count=int(problem.layout.total_qubits),
        source_label=str(problem.reference_state.source_label),
        state_fingerprint=projective_state_fingerprint(reference_array),
    )

    estimator_prefix = _mapping(
        checkpoint.get("estimator_prefix"),
        label="Append selected estimator prefix",
    )
    executed = _mapping(
        estimator_prefix.get("cumulative_executed_queries"),
        label="Append selected executed queries",
    )
    components_raw = _mapping(
        executed.get("components"),
        label="Append selected work components",
    )
    components = PaperIWorkComponents(
        n_h_outer=_integer(
            components_raw.get("N_H_outer"), label="Append N_H_outer"
        ),
        n_h_refit=_integer(
            components_raw.get("N_H_refit"), label="Append N_H_refit"
        ),
        n_grad=_integer(components_raw.get("N_grad"), label="Append N_grad"),
        n_metric=_integer(
            components_raw.get("N_metric"), label="Append N_metric"
        ),
    )
    s_alg = _integer(executed.get("S_alg"), label="Append selected S_alg")
    if components.s_alg != s_alg:
        raise CommonAccuracyInputError("Append selected S_alg drifted")
    route_contract = _mapping(
        protocol.route_contract,
        label="Append route contract",
    )
    return PaperIPrefixCompileInput(
        source_method="append_adapt",
        controller_round=controller_round,
        active_ansatz_depth=len(labels),
        ordered_operator_labels=labels,
        operators=tuple(operators),
        logical_parameters=logical_parameters,
        runtime_parameters=runtime_parameters,
        reference_state=reference,
        checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        projective_state_fingerprint=str(
            checkpoint["projective_state_fingerprint"]
        ),
        problem_request_sha256=str(protocol.problem.problem_request_sha256),
        route_profile=str(route_contract.get("route_profile", "")),
        route_contract_sha256=str(route_contract.get("sha256", "")),
        algorithmic_work=PaperIAlgorithmicWork(
            components=components,
            s_alg=s_alg,
        ),
    )


def select_full_horizon_common_accuracy(
    ra_trace: Sequence[Mapping[str, Any]],
    adapt_trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select pairwise earliest crossings at a shared attainable error."""

    def normalized(
        trace: Sequence[Mapping[str, Any]],
        *,
        label: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for expected_round, raw in enumerate(trace, start=1):
            row = _mapping(raw, label=f"{label} row")
            controller_round = _integer(
                row.get("controller_round"),
                label=f"{label} controller round",
                minimum=1,
            )
            error = _finite(
                row.get("absolute_energy_error"),
                label=f"{label} absolute error",
                minimum=0.0,
            )
            if controller_round != expected_round:
                raise CommonAccuracyInputError(
                    f"{label} is not contiguous"
                )
            rows.append(
                {
                    "controller_round": controller_round,
                    "absolute_energy_error": error,
                }
            )
        if len(rows) != 50:
            raise CommonAccuracyInputError(
                f"{label} does not span the 50-round horizon"
            )
        return rows

    ra = normalized(ra_trace, label="RA trace")
    adapt = normalized(adapt_trace, label="ADAPT trace")
    ra_minimum = min(float(row["absolute_energy_error"]) for row in ra)
    adapt_minimum = min(
        float(row["absolute_energy_error"]) for row in adapt
    )
    target = max(ra_minimum, adapt_minimum)
    inclusive_target = math.nextafter(target, math.inf)
    ra_crossing = next(
        row
        for row in ra
        if float(row["absolute_energy_error"]) <= inclusive_target
    )
    adapt_crossing = next(
        row
        for row in adapt
        if float(row["absolute_energy_error"]) <= inclusive_target
    )
    limiting_method = (
        "ra"
        if ra_minimum > adapt_minimum
        else "adapt"
        if adapt_minimum > ra_minimum
        else "equal"
    )
    return {
        "policy": (
            "pairwise_full_50_round_first_crossing_at_shared_"
            "attainable_same_cutoff_error_v1"
        ),
        "common_target_absolute_error": target,
        "ra_horizon_minimum_error": ra_minimum,
        "adapt_horizon_minimum_error": adapt_minimum,
        "limiting_method": limiting_method,
        "ra_crossing_controller_round": int(
            ra_crossing["controller_round"]
        ),
        "ra_crossing_absolute_error": float(
            ra_crossing["absolute_energy_error"]
        ),
        "adapt_crossing_controller_round": int(
            adapt_crossing["controller_round"]
        ),
        "adapt_crossing_absolute_error": float(
            adapt_crossing["absolute_energy_error"]
        ),
        "horizon_controller_rounds": 50,
    }


def _dominance(
    ra_costs: Mapping[str, int],
    adapt_costs: Mapping[str, int],
    fields: Sequence[str],
) -> str:
    ra_no_worse = all(ra_costs[field] <= adapt_costs[field] for field in fields)
    adapt_no_worse = all(
        adapt_costs[field] <= ra_costs[field] for field in fields
    )
    ra_strict = any(ra_costs[field] < adapt_costs[field] for field in fields)
    adapt_strict = any(
        adapt_costs[field] < ra_costs[field] for field in fields
    )
    if ra_no_worse and ra_strict:
        return "RA"
    if adapt_no_worse and adapt_strict:
        return "ADAPT"
    if not ra_strict and not adapt_strict:
        return "equal"
    return "mixed"


def classify_costs(
    ra_costs: Mapping[str, int],
    adapt_costs: Mapping[str, int],
) -> dict[str, Any]:
    """Return explicit ADAPT/RA ratios and dominance verdicts."""

    ratios: dict[str, float | str] = {}
    for field in COST_FIELDS:
        ra = _integer(ra_costs[field], label=f"RA {field}")
        adapt = _integer(adapt_costs[field], label=f"ADAPT {field}")
        if ra == 0:
            ratios[field] = "equal_zero" if adapt == 0 else "infinity"
        else:
            ratios[field] = adapt / ra
    s_verdict = (
        "RA"
        if ra_costs["S_alg"] < adapt_costs["S_alg"]
        else "ADAPT"
        if adapt_costs["S_alg"] < ra_costs["S_alg"]
        else "equal"
    )
    return {
        "ratio_definition": "ADAPT_cost_divided_by_RA_cost",
        "ratio_interpretation": (
            "greater_than_one_RA_cheaper_less_than_one_ADAPT_cheaper"
        ),
        "ratios": ratios,
        "circuit_verdict": _dominance(
            ra_costs,
            adapt_costs,
            QISKIT_FIELDS,
        ),
        "s_alg_verdict": s_verdict,
        "overall_verdict": _dominance(
            ra_costs,
            adapt_costs,
            COST_FIELDS,
        ),
    }


def _compile_cost(
    prefix: Any,
    *,
    cache: dict[tuple[str, str, str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    key = prefix.compile_cache_key
    cached = cache.get(key)
    if cached is not None:
        return cached
    qiskit, checkpoint_sha, payload = master._compile_prefix_qiskit(
        prefix,
        compiler=None,
    )
    costs = {
        "N2q": _integer(qiskit["N2q"], label="compiled N2q"),
        "D2q": _integer(qiskit["D2q"], label="compiled D2q"),
        "Dc": _integer(qiskit["Dc"], label="compiled Dc"),
        "W1q": _integer(qiskit["W1q"], label="compiled W1q"),
        "S_alg": _integer(
            prefix.algorithmic_work.s_alg,
            label="prefix S_alg",
        ),
    }
    compiled = {
        "costs": costs,
        "prefix": {
            "source_method": prefix.source_method,
            "controller_round": prefix.controller_round,
            "active_ansatz_depth": prefix.active_ansatz_depth,
            "checkpoint_sha256": checkpoint_sha,
            "problem_request_sha256": prefix.problem_request_sha256,
            "route_profile": prefix.route_profile,
            "route_contract_sha256": prefix.route_contract_sha256,
        },
        "qiskit": {
            "compile_convention": payload.get("compile_convention"),
            "qiskit_version": payload.get("qiskit_version"),
            "compiled_basis_gates": payload.get("compiled_basis_gates"),
            "qiskit_transpile_optimization_level": payload.get(
                "qiskit_transpile_optimization_level"
            ),
            "qiskit_transpile_seed": payload.get("qiskit_transpile_seed"),
            "compiled_circuit_scope": payload.get("compiled_circuit_scope"),
            "qiskit_basis_work_schema": qiskit.get(
                "qiskit_basis_work_schema"
            ),
            "generator_coefficients_sha256": payload.get(
                "generator_coefficients_sha256"
            ),
        },
    }
    cache[key] = compiled
    return compiled


def _compile_cell_prefix(
    cell: Mapping[str, Any],
    *,
    controller_round: int,
    prefix_cache: dict[tuple[str, int], dict[str, Any]],
    compile_cache: dict[
        tuple[str, str, str, str, int],
        dict[str, Any],
    ],
) -> dict[str, Any]:
    key = (str(cell["execution_id"]), controller_round)
    cached = prefix_cache.get(key)
    if cached is not None:
        return cached
    if cell["method_family"] == "ra":
        prefix = _ra_prefix_from_archive(
            cell,
            controller_round=controller_round,
        )
    elif cell["method_family"] == "append":
        master._configure_package_dir(Path(str(cell["package_dir"])))
        prefix = _append_prefix_from_archive(
            cell,
            controller_round=controller_round,
        )
    else:
        raise CommonAccuracyInputError("unknown prefix compilation family")
    compiled = _compile_cost(prefix, cache=compile_cache)
    prefix_cache[key] = compiled
    return compiled


def _build_comparisons(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    append = {
        (str(row["representation"]), str(row["regime"])): row
        for row in cells
        if row["method_family"] == "append"
    }
    ra = {
        (
            str(row["representation"]),
            str(row["regime"]),
            str(row["policy"]),
        ): row
        for row in cells
        if row["method_family"] == "ra"
    }
    comparisons: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    prefix_cache: dict[tuple[str, int], dict[str, Any]] = {}
    requested_prefixes: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for regime, title, abbreviation in REGIMES:
            adapt_cell = append[(representation, regime)]
            for policy in POLICIES:
                ra_cell = ra.get((representation, regime, policy))
                if ra_cell is None:
                    comparisons.append(
                        {
                            "representation": representation,
                            "regime": regime,
                            "regime_title": title,
                            "abbreviation": abbreviation,
                            "ra_policy": policy,
                            "status": "unavailable",
                            "reason": "validated_RA_cell_unavailable",
                            "adapt_execution_id": adapt_cell["execution_id"],
                            "ra_execution_id": None,
                        }
                    )
                    continue
                if not math.isclose(
                    float(ra_cell["exact_same_cutoff_energy"]),
                    float(adapt_cell["exact_same_cutoff_energy"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-10,
                ):
                    raise CommonAccuracyInputError(
                        f"{representation} {regime} {policy}: exact "
                        "references disagree"
                    )
                selection = select_full_horizon_common_accuracy(
                    _sequence(ra_cell["trace"], label="RA trace"),
                    _sequence(adapt_cell["trace"], label="ADAPT trace"),
                )
                requested_prefixes.append(
                    {
                        "representation": representation,
                        "regime": regime,
                        "ra_policy": policy,
                        "ra_controller_round": selection[
                            "ra_crossing_controller_round"
                        ],
                        "adapt_controller_round": selection[
                            "adapt_crossing_controller_round"
                        ],
                    }
                )
                print(
                    (
                        f"compile {representation} {regime} {policy}: "
                        f"RA k={selection['ra_crossing_controller_round']}, "
                        "ADAPT k="
                        f"{selection['adapt_crossing_controller_round']}"
                    ),
                    flush=True,
                )
                ra_compiled = _compile_cell_prefix(
                    ra_cell,
                    controller_round=selection[
                        "ra_crossing_controller_round"
                    ],
                    prefix_cache=prefix_cache,
                    compile_cache=cache,
                )
                adapt_compiled = _compile_cell_prefix(
                    adapt_cell,
                    controller_round=selection[
                        "adapt_crossing_controller_round"
                    ],
                    prefix_cache=prefix_cache,
                    compile_cache=cache,
                )
                cost_comparison = classify_costs(
                    _mapping(ra_compiled["costs"], label="RA costs"),
                    _mapping(adapt_compiled["costs"], label="ADAPT costs"),
                )
                comparisons.append(
                    {
                        "representation": representation,
                        "regime": regime,
                        "regime_title": title,
                        "abbreviation": abbreviation,
                        "ra_policy": policy,
                        "status": "compared",
                        "same_cutoff_exact_energy": ra_cell[
                            "exact_same_cutoff_energy"
                        ],
                        "selection": selection,
                        "ra_execution_id": ra_cell["execution_id"],
                        "adapt_execution_id": adapt_cell["execution_id"],
                        "ra_source": ra_cell["source"],
                        "adapt_source": adapt_cell["source"],
                        "ra_compiled": ra_compiled,
                        "adapt_compiled": adapt_compiled,
                        "cost_comparison": cost_comparison,
                    }
                )
    return comparisons, {
        "requested_prefix_count": 2
        * sum(row["status"] == "compared" for row in comparisons),
        "unique_compiled_prefix_count": len(cache),
        "unique_source_prefix_count": len(prefix_cache),
        "requested_prefixes": requested_prefixes,
    }


def _format_s_alg(value: int) -> str:
    if value == 0:
        return "0.0e0"
    exponent = int(math.floor(math.log10(value)))
    coefficient = value / (10**exponent)
    if round(coefficient, 1) >= 10.0:
        coefficient /= 10.0
        exponent += 1
    return f"{coefficient:.1f}e{exponent}"


def _cost_tuple_tex(costs: Mapping[str, Any] | None) -> str:
    if costs is None:
        return r"\textemdash"
    return (
        r"$("
        + ",".join(
            (
                str(costs["N2q"]),
                str(costs["D2q"]),
                str(costs["Dc"]),
                str(costs["W1q"]),
                _format_s_alg(int(costs["S_alg"])),
            )
        )
        + r")$"
    )


def _ratio_tuple_tex(
    comparison: Mapping[str, Any] | None,
) -> str:
    if comparison is None:
        return r"\textemdash"
    ratios = _mapping(comparison.get("ratios"), label="cost ratios")
    values: list[str] = []
    for field in COST_FIELDS:
        value = ratios[field]
        if value == "infinity":
            values.append(r"\infty")
        elif value == "equal_zero":
            values.append("1.00")
        else:
            values.append(f"{float(value):.2f}")
    return r"$(" + ",".join(values) + r")$"


def _error_tex(value: float | None) -> str:
    if value is None:
        return r"\textemdash"
    if value == 0.0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(value))))
    coefficient = value / (10**exponent)
    return rf"${coefficient:.2f}\!\times\!10^{{{exponent}}}$"


def _tex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _verdict_tex(value: str) -> str:
    if value == "RA":
        return r"\textcolor{blue!70!black}{\textbf{RA}}"
    if value == "ADAPT":
        return r"\textcolor{orange!80!black}{\textbf{ADAPT}}"
    if value == "equal":
        return "equal"
    return "mixed"


def _table_tex(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    representation: str,
) -> str:
    selected = [
        row for row in comparisons if row["representation"] == representation
    ]
    lines = [
        r"\begin{center}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\begin{tabular}{@{}llrrp{3.0in}p{2.35in}p{1.25in}@{}}",
        r"\toprule",
        (
            r"Reg. & RA policy & $\epsilon_\star$ & $(k_{\rm RA},k_{\rm A})$ "
            r"& Crossing-prefix costs & $C_{\rm A}/C_{\rm RA}$ "
            r"& Winners \\"
        ),
        r"\midrule",
    ]
    previous_regime: str | None = None
    for row in selected:
        regime = str(row["regime"])
        if previous_regime is not None and regime != previous_regime:
            lines.append(r"\addlinespace[0.3ex]")
        previous_regime = regime
        policy_label = (
            "none" if row["ra_policy"] == "no_insertion" else "plateau"
        )
        if row["status"] != "compared":
            lines.append(
                " & ".join(
                    (
                        _tex_escape(row["abbreviation"]),
                        _tex_escape(policy_label),
                        r"\multicolumn{5}{l}{"
                        r"\textcolor{red!65!black}{unavailable: validated RA "
                        r"cell did not complete}}",
                    )
                )
                + r" \\"
            )
            continue
        selection = _mapping(row["selection"], label="selection")
        ra_costs = _mapping(
            _mapping(row["ra_compiled"], label="RA compiled").get("costs"),
            label="RA costs",
        )
        adapt_costs = _mapping(
            _mapping(
                row["adapt_compiled"], label="ADAPT compiled"
            ).get("costs"),
            label="ADAPT costs",
        )
        cost_comparison = _mapping(
            row["cost_comparison"],
            label="cost comparison",
        )
        lines.append(
            " & ".join(
                (
                    _tex_escape(row["abbreviation"]),
                    _tex_escape(policy_label),
                    _error_tex(
                        float(selection["common_target_absolute_error"])
                    ),
                    (
                        "$("
                        + str(selection["ra_crossing_controller_round"])
                        + ","
                        + str(selection["adapt_crossing_controller_round"])
                        + ")$"
                    ),
                    (
                        r"\shortstack[l]{RA: "
                        + _cost_tuple_tex(ra_costs)
                        + r"\\ADAPT: "
                        + _cost_tuple_tex(adapt_costs)
                        + "}"
                    ),
                    _ratio_tuple_tex(cost_comparison),
                    (
                        r"\shortstack[l]{circuit: "
                        + _verdict_tex(
                            str(cost_comparison["circuit_verdict"])
                        )
                        + r"\\$S_{\rm alg}$: "
                        + _verdict_tex(
                            str(cost_comparison["s_alg_verdict"])
                        )
                        + r"\\overall: "
                        + _verdict_tex(
                            str(cost_comparison["overall_verdict"])
                        )
                        + "}"
                    ),
                )
            )
            + r" \\"
        )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        )
    )
    return "\n".join(lines)


def _score_counts(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    representation: str,
) -> str:
    counts = {"RA": 0, "ADAPT": 0, "mixed": 0, "equal": 0}
    unavailable = 0
    for row in comparisons:
        if row["representation"] != representation:
            continue
        if row["status"] != "compared":
            unavailable += 1
            continue
        verdict = str(
            _mapping(
                row["cost_comparison"], label="cost comparison"
            )["overall_verdict"]
        )
        counts[verdict] += 1
    return (
        f"overall dominance: RA={counts['RA']}, ADAPT={counts['ADAPT']}, "
        f"mixed={counts['mixed']}, equal={counts['equal']}; "
        f"unavailable={unavailable}"
    )


def _write_tex(
    *,
    comparisons: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> Path:
    tex = OUTPUT_DIR / f"{STEM}.tex"
    manifest = _mapping(
        source.get("parameter_manifest"),
        label="parameter manifest",
    )
    page = r"""
\begin{center}
{\Large\bfseries %s}\\[0.2ex]
{\fontsize{8.2}{9.2}\selectfont Pairwise full-horizon common-attainable-error
costs; same-cutoff exact reference.}
\end{center}
\vspace{0.5ex}
%s
\vspace{1.0ex}
{\fontsize{6.05}{6.75}\selectfont
\setlength{\tabcolsep}{2.5pt}
%s
}
\vfill
\fcolorbox{black!30}{black!2}{\begin{minipage}{0.975\textwidth}
\fontsize{6.55}{7.4}\selectfont
\textbf{How to read the table.}
$\epsilon_\star=\max(\min_{k\leq50}\epsilon_{\rm RA},
\min_{k\leq50}\epsilon_{\rm ADAPT})$; each method uses its earliest crossing.
$C=(N_{2q},D_{2q},D_c,W_{1q},S_{\rm alg})$ at that authenticated prefix.
The ratio vector is $C_{\rm ADAPT}/C_{\rm RA}$: values above one mean RA is
cheaper; values below one mean ADAPT is cheaper. ``Overall'' names a winner
only under five-coordinate dominance; otherwise it says mixed.
\end{minipage}}
"""
    manifest_block = (
        r"\fcolorbox{black!30}{black!2}{\begin{minipage}{0.975\textwidth}"
        r"\fontsize{6.55}{7.4}\selectfont "
        r"\textbf{Parameter manifest.} "
        + _tex_escape(
            "Hubbard--Holstein; L=2; open boundary; half-filled sector; "
            "binary bosons; nph=3 for weak phonon coupling and nph=7 for "
            "strong phonon coupling; exact diagonalization at the identical "
            "cutoff. "
            f"Horizon={manifest['horizon']}; optimizer="
            f"{manifest['optimizer']}-{manifest['optimizer_maxiter']}; "
            f"seed={_mapping(manifest['seeds'], label='seeds')['adapt']}; "
            f"RA active gradients={manifest['active_gradient_policy']}. "
            "Conventional comparator=unwhitened ADAPT."
        )
        + r"\par "
        + _tex_escape(
            "No terminal or plateau cost substitution is permitted. "
            "Every displayed circuit cost is freshly compiled from the "
            "authenticated first-crossing prefix through the shared locked "
            "Paper-I Qiskit path."
        )
        + r"\end{minipage}}"
    )
    macro_status = (
        r"\begin{center}{\fontsize{7.0}{8.0}\selectfont "
        + _tex_escape(
            _score_counts(comparisons, representation="macro")
        )
        + r"}\end{center}"
    )
    singleton_status = _tex_escape(
        _score_counts(comparisons, representation="singleton")
    )
    body = (
        r"\documentclass[letterpaper,landscape]{article}" "\n"
        r"\usepackage[margin=0.30in]{geometry}" "\n"
        r"\usepackage{booktabs}" "\n"
        r"\usepackage{xcolor}" "\n"
        r"\pagestyle{empty}" "\n"
        r"\setlength{\parindent}{0pt}" "\n"
        r"\begin{document}" "\n"
        + (
            page
            % (
                "Macro RA versus conventional ADAPT",
                manifest_block + r"\par\vspace{0.4ex}" + macro_status,
                _table_tex(comparisons, representation="macro"),
            )
        )
        + r"\clearpage"
        + "\n"
        + (
            page
            % (
                "Singleton RA versus conventional ADAPT",
                (
                    r"\begin{center}{\fontsize{7.0}{8.0}\selectfont "
                    r"Same parameter, source-lock, target, and compilation "
                    r"contracts as page 1.\par "
                    + singleton_status
                    + r"}\end{center}"
                ),
                _table_tex(comparisons, representation="singleton"),
            )
        )
        + r"\end{document}"
        + "\n"
    )
    tex.write_text(body, encoding="utf-8")
    return tex


def _compile_tex(tex: Path) -> tuple[Path, dict[str, Any]]:
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    build_dir = REPO_ROOT / "tmp/pdfs" / tex.stem
    build_dir.mkdir(parents=True, exist_ok=True)
    if latexmk:
        command = [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={build_dir}",
            tex.name,
        ]
    elif pdflatex:
        command = [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={build_dir}",
            tex.name,
        ]
    else:
        raise RuntimeError("latexmk or pdflatex is required")
    completed = subprocess.run(
        command,
        cwd=tex.parent,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "FORCE_SOURCE_DATE": "1",
            "SOURCE_DATE_EPOCH": "1785283200",
            "TZ": "UTC",
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "LaTeX build failed:\n"
            + completed.stdout[-5000:]
            + completed.stderr[-5000:]
        )
    compiled = build_dir / f"{tex.stem}.pdf"
    if not compiled.is_file():
        raise RuntimeError("LaTeX completed without producing a PDF")
    destination = tex.with_suffix(".pdf")
    shutil.copy2(compiled, destination)
    log = build_dir / f"{tex.stem}.log"
    log_text = log.read_text(encoding="utf-8", errors="replace")
    return destination, {
        "engine": Path(command[0]).name,
        "returncode": completed.returncode,
        "overfull_hbox_count": log_text.count("Overfull \\hbox"),
        "underfull_hbox_count": log_text.count("Underfull \\hbox"),
        "fatal_error_present": "!  ==> Fatal error occurred" in log_text,
    }


def build() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cells, source = _load_cells()
    comparisons, compilation = _build_comparisons(cells)
    tex = _write_tex(comparisons=comparisons, source=source)
    pdf, latex_validation = _compile_tex(tex)
    if pdf.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("generated file lacks a PDF header")
    try:
        from pypdf import PdfReader

        page_count = len(PdfReader(str(pdf)).pages)
    except Exception as exc:
        raise RuntimeError(
            f"generated PDF structural read failed: {exc}"
        ) from exc
    if page_count != 2:
        raise RuntimeError(
            f"generated scorecard has {page_count} pages, expected 2"
        )
    provenance_path = OUTPUT_DIR / f"{STEM}_provenance.json"
    provenance = {
        "schema": "paper_i_ra_vs_adapt_common_attainable_error_cost_v1",
        "status": "diagnostic_comparison_not_paper_evidence",
        "paper_evidence_adopted": False,
        "metric": "same_cutoff_absolute_energy_error",
        "selection_policy": (
            "pairwise_full_50_round_first_crossing_at_shared_"
            "attainable_same_cutoff_error_v1"
        ),
        "cost_tuple": {
            "fields": list(COST_FIELDS),
            "qiskit_fields": list(QISKIT_FIELDS),
            "s_alg_display_notation": "X.YeZ_two_significant_digits",
            "prefix_scope": "earliest_common_target_crossing",
        },
        "ratio": {
            "definition": "ADAPT_cost_divided_by_RA_cost",
            "greater_than_one": "RA_cheaper",
            "less_than_one": "ADAPT_cheaper",
        },
        "source": source,
        "cell_count_consumed": len(cells),
        "comparison_row_count": len(comparisons),
        "compared_row_count": sum(
            row["status"] == "compared" for row in comparisons
        ),
        "unavailable_row_count": sum(
            row["status"] != "compared" for row in comparisons
        ),
        "compilation": compilation,
        "rows": comparisons,
        "limitations": [
            (
                "Four RA plateau cells are unavailable because their "
                "validated attempts did not complete; no substitute is used."
            ),
            (
                "Corrected commutation-reduced RA-always is pending and is "
                "outside this first scorecard."
            ),
            (
                "Pairwise targets make each RA policy fair against ADAPT, "
                "but targets can differ between RA policies."
            ),
            (
                "This diagnostic report does not promote, replace, or demote "
                "Paper-I evidence."
            ),
        ],
        "validation": {
            "page_count": page_count,
            "pdf_header_valid": True,
            "latex": latex_validation,
            "visual_inspection": "pending",
        },
        "outputs": {
            "pdf": {
                "path": str(pdf),
                "sha256": _sha256_file(pdf),
                "size_bytes": pdf.stat().st_size,
            },
            "tex": {
                "path": str(tex),
                "sha256": _sha256_file(tex),
            },
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pdf, provenance_path


def main() -> int:
    try:
        pdf, provenance = build()
    except (
        CommonAccuracyInputError,
        master.ReportInputError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2
    print(
        json.dumps(
            {
                "status": "passed",
                "pdf": str(pdf),
                "pdf_sha256": _sha256_file(pdf),
                "provenance": str(provenance),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

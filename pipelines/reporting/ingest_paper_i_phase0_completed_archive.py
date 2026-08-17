#!/usr/bin/env python3
"""Ingest one authenticated Page-12 CHTC full archive for reporting.

The utility does not retrieve, delete, or otherwise mutate remote state.  It
binds a caller-supplied remote archive identity to an exact local copy, checks
the worker/manifest/summary closure inside that archive, recompiles the signed
round-50 prefix through the shared Paper-I compiler, and emits the compact
adapter consumed by :mod:`append_paper_i_phase0_route_pages`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tarfile
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_"
    "qiskit_phase23_no_lanes_cap24_tau1em4_r50_20260807_v1_chtc"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_phase0_completed_20260809"
)
TARGET_ROUND = 50
COMPILE_CONVENTION = "table_i_basis_gate_transpile_v1"
ROUTE_CONTRACT_SHA256 = (
    "9811652b332b592bee048a8e5f3048972256abae186921ed7efea52bfd5f3dd8"
)


class IngestError(ValueError):
    """The completed archive cannot support an authenticated report row."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    if "sha256" in unsigned:
        raise IngestError("refusing to digest a pre-digested mapping")
    return {
        **unsigned,
        "sha256": hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
    }


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> None:
    claimed = value.get("sha256")
    unsigned = {key: row for key, row in value.items() if key != "sha256"}
    observed = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise IngestError(f"{label}: self digest drifted")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IngestError(f"{label}: mapping required")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise IngestError(f"{label}: sequence required")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise IngestError(f"{label}: integer required")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise IngestError(f"{label}: integer required") from exc
    if result != value or result < minimum:
        raise IngestError(f"{label}: invalid integer")
    return result


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise IngestError(f"{label}: finite scalar required") from exc
    if not math.isfinite(result):
        raise IngestError(f"{label}: finite scalar required")
    return result


def _safe_member_name(raw: str) -> str:
    name = str(raw)
    while name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise IngestError(f"unsafe archive member: {raw!r}")
    return path.as_posix()


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IngestError(f"{label}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise IngestError(f"{label}: JSON object required")
    return value


def _remote_binding(
    *, path: str, sha256: str, size_bytes: int
) -> dict[str, Any]:
    remote_path = str(path)
    remote_sha = str(sha256).lower()
    if not remote_path or len(remote_sha) != 64 or any(
        character not in "0123456789abcdef" for character in remote_sha
    ):
        raise IngestError("remote archive binding is invalid")
    return {
        "path": remote_path,
        "sha256": remote_sha,
        "size_bytes": _integer(
            size_bytes, label="remote archive size", minimum=1
        ),
    }


def _relative_local_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _job_for_proc(package_dir: Path, proc_id: int) -> tuple[Path, dict[str, Any]]:
    queue_path = package_dir / "queue.tsv"
    if not queue_path.is_file() or queue_path.is_symlink():
        raise IngestError("Page-12 queue.tsv is unavailable")
    rows = [line.split("\t") for line in queue_path.read_text().splitlines()]
    proc = _integer(proc_id, label="proc id")
    if proc >= len(rows) or len(rows[proc]) != 8:
        raise IngestError(f"proc {proc}: queue row is unavailable or malformed")
    execution_id, job_relative, _protocol_relative, job_file_sha, *_ = rows[proc]
    job_path = package_dir / job_relative
    if not job_path.is_file() or job_path.is_symlink():
        raise IngestError(f"proc {proc}: job spec is unavailable")
    if sha256_file(job_path) != job_file_sha:
        raise IngestError(f"proc {proc}: queue-bound job bytes drifted")
    job = _json_object(job_path.read_bytes(), label=f"proc {proc} job")
    verify_self_digest(job, label=f"proc {proc} job")
    if (
        job.get("execution_id") != execution_id
        or job.get("target_horizon") != TARGET_ROUND
        or job.get("candidate_representation") != "single_pauli_word_v1"
        or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
    ):
        raise IngestError(f"proc {proc}: Page-12 job identity drifted")
    return job_path, job


def _archive_documents(
    archive_path: Path,
    *,
    execution_id: str,
) -> tuple[dict[str, bytes], int]:
    if not archive_path.is_file() or archive_path.is_symlink():
        raise IngestError(f"unsafe or missing local archive: {archive_path}")
    wanted = {
        "worker_receipt.json",
        f"runs/{execution_id}/execution_manifest.json",
        f"runs/{execution_id}/summary/summary.json",
    }
    found: dict[str, bytes] = {}
    names: set[str] = set()
    file_count = 0
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                name = _safe_member_name(member.name)
                if name in names:
                    raise IngestError(f"duplicate archive member: {name}")
                names.add(name)
                if member.issym() or member.islnk():
                    raise IngestError(f"link archive member is forbidden: {name}")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise IngestError(f"non-regular archive member: {name}")
                file_count += 1
                if name in wanted:
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise IngestError(f"archive member has no bytes: {name}")
                    payload = stream.read()
                    if len(payload) != member.size:
                        raise IngestError(f"archive member was truncated: {name}")
                    found[name] = payload
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise IngestError(f"full archive scan failed: {exc}") from exc
    missing = sorted(wanted - set(found))
    if missing:
        raise IngestError(f"full archive lacks report members: {missing}")
    return found, file_count


def _artifact_binding(
    worker: Mapping[str, Any], *, relative: str, payload: bytes
) -> dict[str, Any]:
    rows = _sequence(worker.get("artifacts"), label="worker artifacts")
    matching = [
        _mapping(row, label="worker artifact")
        for row in rows
        if isinstance(row, Mapping) and row.get("path") == relative
    ]
    if len(matching) != 1:
        raise IngestError(f"worker artifact binding missing: {relative}")
    expected = matching[0]
    observed = {
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
    }
    if (
        expected.get("sha256") != observed["sha256"]
        or expected.get("size_bytes") != observed["size_bytes"]
    ):
        raise IngestError(f"worker artifact bytes drifted: {relative}")
    return observed


def _work(value: Any, *, label: str) -> dict[str, Any]:
    row = _mapping(value, label=label)
    raw = _mapping(row.get("components"), label=f"{label} components")
    keys = ("n_h_outer", "n_h_refit", "n_grad", "n_metric")
    if set(raw) != set(keys):
        raise IngestError(f"{label}: component closure drifted")
    components = {
        key: _integer(raw[key], label=f"{label}.{key}") for key in keys
    }
    s_alg = _integer(row.get("s_alg"), label=f"{label}.s_alg")
    if s_alg != sum(components.values()):
        raise IngestError(f"{label}: S_alg does not close")
    return {"components": components, "s_alg": s_alg}


def _prefix_compile_input(prefix: Mapping[str, Any]) -> Any:
    from pipelines.reporting.paper_i_run_summary import (
        PaperIAlgorithmicWork,
        PaperIPrefixCompileInput,
        PaperIPrefixOperator,
        PaperIPrefixPauliTerm,
        PaperIReferenceState,
        PaperIWorkComponents,
    )

    reference = _mapping(prefix.get("reference_state"), label="round-50 reference")
    work = _work(prefix.get("algorithmic_work"), label="round-50 prefix work")
    operators = []
    for raw_operator in _sequence(prefix.get("operators"), label="round-50 operators"):
        operator = _mapping(raw_operator, label="round-50 operator")
        terms = tuple(
            PaperIPrefixPauliTerm(
                pauli_exyz=str(term["pauli_exyz"]),
                coefficient_real=_finite(
                    term.get("coefficient_real"), label="runtime coefficient real"
                ),
                coefficient_imaginary=_finite(
                    term.get("coefficient_imaginary"),
                    label="runtime coefficient imaginary",
                ),
                qubit_count=_integer(
                    term.get("qubit_count"), label="runtime qubit count", minimum=1
                ),
            )
            for term in (
                _mapping(value, label="runtime term")
                for value in _sequence(
                    operator.get("runtime_terms"), label="runtime terms"
                )
            )
        )
        operators.append(
            PaperIPrefixOperator(
                candidate_label=str(operator.get("candidate_label", "")),
                logical_index=_integer(
                    operator.get("logical_index"), label="logical index"
                ),
                runtime_start=_integer(
                    operator.get("runtime_start"), label="runtime start"
                ),
                runtime_count=_integer(
                    operator.get("runtime_count"),
                    label="runtime count",
                    minimum=1,
                ),
                execution_mode=str(operator.get("execution_mode", "")),
                runtime_terms=terms,
            )
        )
    components = work["components"]
    return PaperIPrefixCompileInput(
        source_method=str(prefix.get("source_method", "")),
        controller_round=_integer(
            prefix.get("controller_round"), label="prefix round", minimum=1
        ),
        active_ansatz_depth=_integer(
            prefix.get("active_ansatz_depth"), label="prefix depth", minimum=1
        ),
        ordered_operator_labels=tuple(
            str(value)
            for value in _sequence(
                prefix.get("ordered_operator_labels"), label="operator labels"
            )
        ),
        operators=tuple(operators),
        logical_parameters=tuple(
            _finite(value, label="logical parameter")
            for value in _sequence(
                prefix.get("logical_parameters"), label="logical parameters"
            )
        ),
        runtime_parameters=tuple(
            _finite(value, label="runtime parameter")
            for value in _sequence(
                prefix.get("runtime_parameters"), label="runtime parameters"
            )
        ),
        reference_state=PaperIReferenceState(
            amplitudes_real=tuple(
                _finite(value, label="reference real amplitude")
                for value in _sequence(
                    reference.get("amplitudes_real"), label="reference real"
                )
            ),
            amplitudes_imaginary=tuple(
                _finite(value, label="reference imaginary amplitude")
                for value in _sequence(
                    reference.get("amplitudes_imaginary"),
                    label="reference imaginary",
                )
            ),
            qubit_count=_integer(
                reference.get("qubit_count"), label="reference qubits", minimum=1
            ),
            source_label=str(reference.get("source_label", "")),
            state_fingerprint=str(reference.get("state_fingerprint", "")),
        ),
        checkpoint_sha256=str(prefix.get("checkpoint_sha256", "")),
        projective_state_fingerprint=str(
            prefix.get("projective_state_fingerprint", "")
        ),
        problem_request_sha256=str(prefix.get("problem_request_sha256", "")),
        route_profile=str(prefix.get("route_profile", "")),
        route_contract_sha256=str(prefix.get("route_contract_sha256", "")),
        algorithmic_work=PaperIAlgorithmicWork(
            components=PaperIWorkComponents(**components),
            s_alg=work["s_alg"],
        ),
    )


def _default_compiler(prefix: Any) -> Mapping[str, Any]:
    from pipelines.reporting.paper_i_run_summary import (
        compile_paper_i_prefix_qiskit_payload,
    )

    return compile_paper_i_prefix_qiskit_payload(prefix)


def _compiled_terminal(
    *,
    prefix: Mapping[str, Any],
    requested: Mapping[str, Any],
    compiler: Callable[[Any], Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, Any]]:
    typed = _prefix_compile_input(prefix)
    if typed.controller_round != TARGET_ROUND:
        raise IngestError("requested prefix is not controller round 50")
    payload = _mapping(compiler(typed), label="shared compiler payload")
    if payload.get("compile_convention") != COMPILE_CONVENTION:
        raise IngestError("round-50 compiler convention drifted")
    if payload.get("qiskit_basis_work_status") != "ok":
        raise IngestError("round-50 Qiskit one-qubit work is unavailable")

    def metric(name: str) -> int:
        value = payload.get(name)
        if value is None:
            raise IngestError(f"compiled metric is absent: {name}")
        return _integer(value, label=f"compiled {name}")

    costs = {
        "N2q": metric("compiled_count_2q_total"),
        "D2q": metric("compiled_depth_2q_total"),
        "Dc": metric("compiled_depth_total"),
        "W1q": metric("qiskit_pretranspile_pauli_1q_work_total"),
        "S_alg": _integer(
            typed.algorithmic_work.s_alg, label="compiled-prefix S_alg"
        ),
    }
    serialized = _mapping(requested.get("resources"), label="summary resources")
    expected = {
        "N2q": serialized.get("compiled_two_qubit_count"),
        "D2q": serialized.get("compiled_two_qubit_depth"),
        "Dc": serialized.get("compiled_total_depth"),
    }
    if (
        serialized.get("compile_convention") != COMPILE_CONVENTION
        or any(costs[key] != expected[key] for key in expected)
    ):
        raise IngestError("shared compiler differs from serialized Qiskit triplet")
    compile_receipt = {
        "compile_convention": payload.get("compile_convention"),
        "qiskit_basis_work_status": payload.get("qiskit_basis_work_status"),
        "qiskit_pretranspile_basis_change_1q_total": _integer(
            payload.get("qiskit_pretranspile_basis_change_1q_total"),
            label="compiled basis-change work",
        ),
        "qiskit_transpile_optimization_level": payload.get(
            "qiskit_transpile_optimization_level"
        ),
        "qiskit_transpile_seed": payload.get("qiskit_transpile_seed"),
        "qiskit_version": payload.get("qiskit_version"),
        "source": "source_locked_PaperIPrefixCompileInput_shared_compiler_cross_checked_v1",
    }
    return costs, compile_receipt


def build_outputs(
    *,
    archive_path: Path,
    cluster_id: int,
    proc_id: int,
    remote_archive: Mapping[str, Any],
    package_dir: Path = DEFAULT_PACKAGE_DIR,
    retrieved_utc: str | None = None,
    compiler: Callable[[Any], Mapping[str, Any]] = _default_compiler,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the self-digested completed adapter and retrieval receipt."""

    cluster = _integer(cluster_id, label="cluster id", minimum=1)
    proc = _integer(proc_id, label="proc id")
    if not archive_path.is_file() or archive_path.is_symlink():
        raise IngestError(f"unsafe or missing local archive: {archive_path}")
    remote = _remote_binding(
        path=str(remote_archive.get("path", "")),
        sha256=str(remote_archive.get("sha256", "")),
        size_bytes=remote_archive.get("size_bytes"),
    )
    if archive_path.stat().st_size != remote["size_bytes"]:
        raise IngestError("local archive size differs from remote identity")
    local_sha = sha256_file(archive_path)
    if local_sha != remote["sha256"]:
        raise IngestError("local archive SHA-256 differs from remote identity")

    job_path, job = _job_for_proc(package_dir, proc)
    execution_id = str(job["execution_id"])
    run_root = f"runs/{execution_id}"
    manifest_relative = f"{run_root}/execution_manifest.json"
    summary_relative = f"{run_root}/summary/summary.json"
    documents, member_count = _archive_documents(
        archive_path, execution_id=execution_id
    )
    worker_payload = documents["worker_receipt.json"]
    manifest_payload = documents[manifest_relative]
    summary_payload = documents[summary_relative]
    worker = _json_object(worker_payload, label="worker receipt")
    manifest = _json_object(manifest_payload, label="execution manifest")
    summary = _json_object(summary_payload, label="run summary")
    verify_self_digest(worker, label="worker receipt")
    verify_self_digest(manifest, label="execution manifest")

    job_sha = str(job["sha256"])
    if (
        worker.get("schema")
        != "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_worker_receipt_v1"
        or worker.get("status") != "passed"
        or worker.get("package_id") != job.get("package_id")
        or worker.get("campaign_id") != job.get("campaign_id")
        or worker.get("execution_id") != execution_id
        or worker.get("job_spec_sha256") != job_sha
        or worker.get("controller_rounds_completed") != TARGET_ROUND
        or worker.get("execution_manifest_sha256") != manifest.get("sha256")
    ):
        raise IngestError("worker receipt identity drifted")
    manifest_binding = _artifact_binding(
        worker, relative=manifest_relative, payload=manifest_payload
    )
    summary_binding = _artifact_binding(
        worker, relative=summary_relative, payload=summary_payload
    )
    manifest_summary = _mapping(
        _mapping(manifest.get("output_payloads"), label="manifest payloads").get(
            "summary"
        ),
        label="manifest summary",
    )
    if (
        manifest.get("schema")
        != "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_execution_manifest_v1"
        or manifest.get("status") != "passed"
        or manifest.get("package_id") != job.get("package_id")
        or manifest.get("campaign_id") != job.get("campaign_id")
        or manifest.get("execution_id") != execution_id
        or manifest.get("job_spec_sha256") != job_sha
        or manifest.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or manifest.get("target_horizon") != TARGET_ROUND
        or manifest.get("controller_rounds_completed") != TARGET_ROUND
        or manifest_summary.get("path") != summary_relative
        or manifest_summary.get("sha256") != summary_binding["sha256"]
        or manifest_summary.get("size_bytes") != summary_binding["size_bytes"]
    ):
        raise IngestError("execution manifest identity drifted")

    provenance = _mapping(summary.get("provenance"), label="summary provenance")
    trace = _sequence(summary.get("accepted_error_trace"), label="accepted trace")
    requested_rows = _sequence(
        summary.get("requested_rounds"), label="requested rounds"
    )
    if (
        summary.get("schema") != "paper_i_run_summary_v1"
        or summary.get("available_controller_rounds") != TARGET_ROUND
        or len(trace) != TARGET_ROUND
        or len(requested_rows) != 1
        or provenance.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or provenance.get("candidate_representation") != "single_pauli_word_v1"
        or provenance.get("qiskit_compile_convention") != COMPILE_CONVENTION
    ):
        raise IngestError("run summary identity drifted")
    exact = _finite(
        provenance.get("exact_same_cutoff_energy"), label="same-cutoff exact energy"
    )
    points: list[dict[str, Any]] = []
    for expected_round, raw in enumerate(trace, 1):
        row = _mapping(raw, label=f"accepted trace row {expected_round}")
        energy = _finite(row.get("accepted_energy"), label="accepted energy")
        error = _finite(row.get("absolute_energy_error"), label="accepted error")
        if (
            row.get("controller_round") != expected_round
            or not math.isclose(
                _finite(row.get("exact_same_cutoff_energy"), label="trace exact"),
                exact,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(error, abs(energy - exact), rel_tol=0.0, abs_tol=1.0e-12)
        ):
            raise IngestError("accepted 50-point trajectory drifted")
        points.append({"k": expected_round, "energy": energy, "error": error})

    requested = _mapping(requested_rows[0], label="round-50 requested observation")
    requested_work = _work(
        requested.get("algorithmic_work"), label="round-50 requested work"
    )
    canonical_work = _work(
        summary.get("canonical_all_work"), label="canonical all work"
    )
    prefix = _mapping(requested.get("prefix"), label="round-50 prefix")
    prefix_work = _work(prefix.get("algorithmic_work"), label="round-50 prefix work")
    if (
        requested.get("status") != "available"
        or requested.get("failure") is not None
        or requested.get("controller_round") != TARGET_ROUND
        or requested_work != canonical_work
        or prefix_work != canonical_work
    ):
        raise IngestError("round-50 work observation drifted")
    costs, compile_receipt = _compiled_terminal(
        prefix=prefix, requested=requested, compiler=compiler
    )
    if costs["S_alg"] != canonical_work["s_alg"]:
        raise IngestError("compiled-prefix S_alg differs from canonical all work")
    terminal = points[-1]
    if not math.isclose(
        _finite(requested.get("absolute_energy_error"), label="requested error"),
        terminal["error"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise IngestError("round-50 requested error differs from trajectory")

    def member_source(relative: str, payload: bytes) -> dict[str, Any]:
        return {
            "path": f"{remote['path']}::{relative}",
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        }

    adapter = digested(
        {
            "schema": "paper_i_phase0_completed_remote_summary_adapter_v1",
            "status": "passed_remote_summary_extract_full_archive_preserved",
            "cluster_id": cluster,
            "proc_id": proc,
            "execution_id": execution_id,
            "regime_id": str(job["regime_id"]),
            "nph": _integer(job.get("nph"), label="job nph", minimum=1),
            "controller_rounds_completed": TARGET_ROUND,
            "exact_same_cutoff_energy": exact,
            "points": points,
            "terminal": {
                "k": TARGET_ROUND,
                "energy": terminal["energy"],
                "error": terminal["error"],
                "costs": costs,
                "compile": compile_receipt,
                "work_components": canonical_work["components"],
            },
            "source": {
                "full_archive": dict(remote),
                "full_archive_local_state": (
                    "verified_local_copy_bound_to_remote_sha256_and_size"
                ),
                "execution_manifest": member_source(
                    manifest_relative, manifest_payload
                ),
                "execution_manifest_canonical_sha256": manifest["sha256"],
                "summary": member_source(summary_relative, summary_payload),
                "worker_receipt": member_source(
                    "worker_receipt.json", worker_payload
                ),
                "worker_receipt_canonical_sha256": worker["sha256"],
                "job_spec": {
                    "path": _relative_local_path(job_path),
                    "canonical_sha256": job_sha,
                    "file_sha256": sha256_file(job_path),
                },
            },
        }
    )
    timestamp = retrieved_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    receipt = digested(
        {
            "schema": "paper_i_chtc_verified_retrieval_receipt_v1",
            "status": "passed",
            "cluster_id": cluster,
            "proc_id": proc,
            "execution_id": execution_id,
            "retrieved_utc": str(timestamp),
            "byte_identity_passed": True,
            "local_archive": {
                "path": _relative_local_path(archive_path),
                "sha256": local_sha,
                "size_bytes": archive_path.stat().st_size,
                "gzip_test_passed": True,
                "full_tar_inventory_scan_passed": True,
                "regular_member_count": member_count,
            },
            "remote_archive": dict(remote),
        }
    )
    return adapter, receipt


def write_outputs(
    *,
    adapter: Mapping[str, Any],
    receipt: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{int(adapter['cluster_id'])}.{int(adapter['proc_id'])}"
    adapter_path = output_dir / f"{stem}_completed_report_adapter.json"
    receipt_path = output_dir / f"{stem}_retrieval_receipt.json"
    for path in (adapter_path, receipt_path):
        if path.exists() or path.is_symlink():
            raise IngestError(f"refusing to overwrite completed output: {path}")
    for path, value in ((adapter_path, adapter), (receipt_path, receipt)):
        with path.open("xb") as stream:
            stream.write(canonical_json_bytes(value) + b"\n")
    return adapter_path, receipt_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--proc-id", type=int, required=True)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--remote-sha256", required=True)
    parser.add_argument("--remote-size-bytes", type=int, required=True)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--retrieved-utc")
    args = parser.parse_args()
    try:
        adapter, receipt = build_outputs(
            archive_path=args.archive.resolve(),
            cluster_id=args.cluster_id,
            proc_id=args.proc_id,
            remote_archive={
                "path": args.remote_path,
                "sha256": args.remote_sha256,
                "size_bytes": args.remote_size_bytes,
            },
            package_dir=args.package_dir.resolve(),
            retrieved_utc=args.retrieved_utc,
        )
        adapter_path, receipt_path = write_outputs(
            adapter=adapter,
            receipt=receipt,
            output_dir=args.output_dir.resolve(),
        )
    except (IngestError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "passed",
                "adapter": str(adapter_path),
                "retrieval_receipt": str(receipt_path),
                "adapter_sha256": adapter["sha256"],
                "retrieval_receipt_sha256": receipt["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

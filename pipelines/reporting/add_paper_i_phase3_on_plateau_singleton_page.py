#!/usr/bin/env python3
"""Append the six-regime Phase-III-on-plateau singleton comparison page.

The input is exactly one authenticated v3 CHTC attempt archive for each of
the six Hubbard--Holstein regimes.  The builder validates the archive member
closure, scheduler attempt receipt, package/activation authorities, worker
receipt, execution manifest, canonical Paper-I summary, and round-50 cost
observation before rendering anything.  The authenticated Append-ADAPT R70
adapter is cropped to the common round-50 horizon and supplies the comparator
trajectory and cost tuple.  The builder then appends one LaTeX-built page while
preserving every existing PDF page at the content-stream level.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v3_chtc"
)
PACKAGE_ID = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v3_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_v1"
)
ROUTE_CONTRACT_SHA256 = (
    "ac868db4dab4f8446ff06e768c5ea77512ef70764efd5699621bd95ad341599d"
)
ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__stationary_source_response_v1__"
    "late_resource_weighting_v1__phase3_population_on_insertion_plateau_v1"
)
PLATEAU_RATIO = 1.0e-4
PLATEAU_COMPARISON = "marginal_to_prior_mean_strictly_below_v2"
PLATEAU_TRIGGER = (
    "immediately_preceding_marginal_over_prior_mean_"
    "accepted_post_full_refit_energy_decrease_v2"
)

REGIME_ORDER = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
NPH_BY_REGIME = {
    "weak_weak": 3,
    "intermediate_weak": 3,
    "strong_weak_u8": 3,
    "weak_strong": 7,
    "intermediate_strong": 7,
    "strong_strong_u8": 7,
}
REGIME_LABELS = {
    "weak_weak": "Weak--weak",
    "intermediate_weak": "Intermediate--weak",
    "strong_weak_u8": "Strong--weak",
    "weak_strong": "Weak--strong",
    "intermediate_strong": "Intermediate--strong",
    "strong_strong_u8": "Strong--strong",
}

ATTEMPT_SCHEMA = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_"
    "worker_attempt_v2"
)
JOB_SCHEMA = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_job_v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_"
    "execution_authorization_v1"
)
EXECUTION_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_"
    "execution_manifest_v2"
)
WORKER_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_"
    "worker_receipt_v2"
)
PACKAGE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_"
    "package_manifest_v1"
)
SUMMARY_SCHEMA = "paper_i_run_summary_v1"
RESULT_SCHEMA = "paper_i_ra_adapt_result_v1"
BASE_ADAPTER_SCHEMA = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_"
    "page8_adapter_v1"
)
ADAPTER_SCHEMA = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_"
    "page8_comparison_adapter_v2"
)
APPEND_ADAPTER_SCHEMA = "paper_i_append_adapt_singleton_r70_progress_adapter_v1"
APPEND_PACKAGE_ID = "paper_i_append_adapt_stationary_core12_r70_fresh_20260731_v1_chtc"
PAGE_REPORT_SCHEMA = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_"
    "page8_comparison_report_v2"
)
BASE_PAGE_ID = "ra_singleton_phase3_population_on_insertion_plateau_r50_v1"
PAGE_ID = (
    "ra_singleton_phase3_population_on_insertion_plateau_"
    "vs_append_r50_v2"
)
REPORT_KEY = "phase3_on_plateau_singleton_sixregime_r50"
COMPILE_CONVENTION = "table_i_basis_gate_transpile_v1"
TARGET_ROUND = 50
APPEND_TRAJECTORY_ROUND = 70
PLOT_FLOOR = 1.0e-16
DEFAULT_APPEND_ADAPTER = REPO_ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "append_singleton_r70_all6_adapter.json"
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ESTIMATOR_SIDECAR = re.compile(
    r"^checkpoint\.estimator_call_ledger_checkpoint\.([0-9a-f]{16})\.json$"
)
_RESUME_SIDECAR = re.compile(
    r"^checkpoint\.verified_singleton_resume\.([0-9a-f]{16})\.json$"
)
_CAPTURE_LIMIT = 96 * 1024 * 1024


class Page8InputError(ValueError):
    """Raised when an input cannot support the guarded page-8 update."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop("sha256", None)
    payload["sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def verify_self_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, Mapping):
        raise Page8InputError(f"{label} must be an object")
    unsigned = copy.deepcopy(dict(value))
    observed = unsigned.pop("sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if observed != expected:
        raise Page8InputError(f"{label} self-digest drifted")
    return expected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Page8InputError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise Page8InputError(f"{label} must be an array")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Page8InputError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Page8InputError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise Page8InputError(f"{label} is outside its finite range")
    return result


def _load_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Page8InputError(f"{label} is unreadable") from exc
    return dict(_mapping(value, label=label))


def _load_json_file(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Page8InputError(f"{label} is unavailable or unsafe")
    try:
        return _load_json_bytes(path.read_bytes(), label=label)
    except OSError as exc:
        raise Page8InputError(f"{label} is unreadable") from exc


def expected_execution_id(regime: str) -> str:
    if regime not in NPH_BY_REGIME:
        raise Page8InputError(f"unsupported regime: {regime}")
    return (
        f"phase3_on_plateau_r50__{regime}__nph{NPH_BY_REGIME[regime]}__"
        "ra_singleton_plateau"
    )


def _safe_member_name(raw: str) -> str:
    pure = PurePosixPath(raw)
    if (
        not raw
        or raw.startswith("/")
        or pure.as_posix() != raw
        or "." in pure.parts
        or ".." in pure.parts
        or any(not part for part in pure.parts)
    ):
        raise Page8InputError(f"unsafe archive member: {raw!r}")
    return raw


class _DigestingReader:
    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()
        self.size = 0
        self.tail = b""

    def read(self, size: int = -1) -> bytes:
        data = self._stream.read(size)
        if data:
            self._digest.update(data)
            self.size += len(data)
            self.tail = (self.tail + data)[-8192:]
        return data

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def drain(self) -> None:
        for _block in iter(lambda: self.read(8 * 1024 * 1024), b""):
            pass


def _capture(reader: _DigestingReader, *, label: str) -> bytes:
    blocks: list[bytes] = []
    total = 0
    while True:
        block = reader.read(1024 * 1024)
        if not block:
            break
        total += len(block)
        if total > _CAPTURE_LIMIT:
            raise Page8InputError(f"{label} exceeds the guarded capture limit")
        blocks.append(block)
    return b"".join(blocks)


def _result_initial_energy(reader: _DigestingReader) -> float:
    try:
        import ijson

        initial: float | None = None
        schema: str | None = None
        for prefix, event, value in ijson.parse(reader, use_float=True):
            if (
                prefix == "run.accepted_transitions.item.energy_before"
                and event == "number"
                and initial is None
            ):
                initial = _finite(value, label="result initial energy")
            elif prefix == "schema" and event == "string":
                schema = str(value)
            if initial is not None and schema is not None:
                break
        reader.drain()
    except (OSError, EOFError, ValueError) as exc:
        raise Page8InputError("result JSON stream is unreadable") from exc
    if initial is None:
        raise Page8InputError("result has no initial accepted-transition energy")
    if schema != RESULT_SCHEMA:
        raise Page8InputError("result schema drifted")
    return initial


def _read_attempt_archive(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Page8InputError(f"attempt archive is unavailable or unsafe: {path}")
    capture_names = {
        "worker_attempt_receipt.json",
        "authority/job.json",
        "authority/execution_authorization.json",
        "authority/activation_manifest.json",
        "worker_outputs/worker_receipt.json",
        "worker_outputs/artifacts/execution_manifest.json",
        "worker_outputs/artifacts/paper_i_summary.json",
    }
    observed: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    initial_energy: float | None = None
    try:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                name = _safe_member_name(member.name)
                if name in observed:
                    raise Page8InputError(f"duplicate archive member: {name}")
                if not member.isfile() or member.issym() or member.islnk():
                    raise Page8InputError(f"unsafe archive member type: {name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise Page8InputError(f"unreadable archive member: {name}")
                reader = _DigestingReader(stream)
                if name == "worker_outputs/artifacts/result.json":
                    initial_energy = _result_initial_energy(reader)
                elif name in capture_names:
                    raw = _capture(reader, label=name)
                    payloads[name] = _load_json_bytes(raw, label=name)
                else:
                    reader.drain()
                if reader.size != member.size:
                    raise Page8InputError(f"archive member size drifted: {name}")
                observed[name] = {
                    "sha256": reader.hexdigest,
                    "size_bytes": reader.size,
                }
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise Page8InputError(f"attempt archive is unreadable: {path}") from exc
    if initial_energy is None:
        raise Page8InputError("attempt archive has no result payload")
    return {
        "archive": file_binding(path),
        "members": observed,
        "payloads": payloads,
        "initial_energy": initial_energy,
    }


def _binding_matches(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        observed.get("sha256") == expected.get("sha256")
        and observed.get("size_bytes") == expected.get("size_bytes")
    )


def _load_package_authority(package_dir: Path) -> dict[str, Any]:
    manifest_path = package_dir / "package_manifest.json"
    manifest = _load_json_file(manifest_path, label="v3 package manifest")
    canonical = verify_self_digest(manifest, label="v3 package manifest")
    expected_ids = tuple(expected_execution_id(regime) for regime in REGIME_ORDER)
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("status") != "passed_inert_six_rows"
        or manifest.get("row_count") != 6
        or tuple(manifest.get("execution_ids", ())) != expected_ids
        or manifest.get("target_horizon") != TARGET_ROUND
    ):
        raise Page8InputError("v3 package identity drifted")
    jobs = {
        str(_mapping(row, label="package job").get("execution_id")): dict(row)
        for row in _sequence(manifest.get("jobs"), label="package jobs")
    }
    protocols = {
        str(_mapping(row, label="package protocol").get("execution_id")): dict(row)
        for row in _sequence(manifest.get("protocols"), label="package protocols")
    }
    if set(jobs) != set(expected_ids) or set(protocols) != set(expected_ids):
        raise Page8InputError("v3 package job/protocol closure drifted")
    source_archive = dict(
        _mapping(manifest.get("source_archive"), label="package source archive")
    )
    return {
        "dir": package_dir,
        "manifest": manifest,
        "manifest_binding": {
            **file_binding(manifest_path),
            "canonical_sha256": canonical,
        },
        "jobs": jobs,
        "protocols": protocols,
        "source_archive": source_archive,
    }


def _work(value: Any, *, label: str) -> dict[str, Any]:
    row = dict(_mapping(value, label=label))
    components = dict(_mapping(row.get("components"), label=f"{label} components"))
    expected = {"n_h_outer", "n_h_refit", "n_grad", "n_metric"}
    if set(components) != expected:
        raise Page8InputError(f"{label} component closure drifted")
    normalized = {
        key: _integer(components[key], label=f"{label}.{key}") for key in expected
    }
    s_alg = _integer(row.get("s_alg"), label=f"{label}.s_alg")
    if s_alg != sum(normalized.values()):
        raise Page8InputError(f"{label} S_alg does not close")
    return {"components": normalized, "s_alg": s_alg}


def _compile_prefix_mapping(prefix: Mapping[str, Any]) -> Mapping[str, Any]:
    from pipelines.reporting.paper_i_run_summary import (
        PaperIAlgorithmicWork,
        PaperIPrefixCompileInput,
        PaperIPrefixOperator,
        PaperIPrefixPauliTerm,
        PaperIReferenceState,
        PaperIWorkComponents,
        compile_paper_i_prefix_qiskit_payload,
    )

    reference = _mapping(prefix.get("reference_state"), label="round-50 reference")
    work = _work(prefix.get("algorithmic_work"), label="round-50 prefix work")
    components = work["components"]
    operators = []
    for raw_operator in _sequence(prefix.get("operators"), label="round-50 operators"):
        operator = _mapping(raw_operator, label="round-50 operator")
        terms = tuple(
            PaperIPrefixPauliTerm(
                pauli_exyz=str(_mapping(raw, label="runtime term")["pauli_exyz"]),
                coefficient_real=_finite(
                    _mapping(raw, label="runtime term").get("coefficient_real"),
                    label="runtime coefficient real",
                ),
                coefficient_imaginary=_finite(
                    _mapping(raw, label="runtime term").get("coefficient_imaginary"),
                    label="runtime coefficient imaginary",
                ),
                qubit_count=_integer(
                    _mapping(raw, label="runtime term").get("qubit_count"),
                    label="runtime qubit count",
                    minimum=1,
                ),
            )
            for raw in _sequence(operator.get("runtime_terms"), label="runtime terms")
        )
        operators.append(
            PaperIPrefixOperator(
                candidate_label=str(operator.get("candidate_label", "")),
                logical_index=_integer(operator.get("logical_index"), label="logical index"),
                runtime_start=_integer(operator.get("runtime_start"), label="runtime start"),
                runtime_count=_integer(
                    operator.get("runtime_count"), label="runtime count", minimum=1
                ),
                execution_mode=str(operator.get("execution_mode", "")),
                runtime_terms=terms,
            )
        )
    typed = PaperIPrefixCompileInput(
        source_method=str(prefix.get("source_method", "")),
        controller_round=_integer(
            prefix.get("controller_round"), label="prefix controller round", minimum=1
        ),
        active_ansatz_depth=_integer(
            prefix.get("active_ansatz_depth"), label="prefix active depth", minimum=1
        ),
        ordered_operator_labels=tuple(
            str(value)
            for value in _sequence(
                prefix.get("ordered_operator_labels"), label="ordered operator labels"
            )
        ),
        operators=tuple(operators),
        logical_parameters=tuple(
            _finite(value, label="logical parameter")
            for value in _sequence(prefix.get("logical_parameters"), label="logical parameters")
        ),
        runtime_parameters=tuple(
            _finite(value, label="runtime parameter")
            for value in _sequence(prefix.get("runtime_parameters"), label="runtime parameters")
        ),
        reference_state=PaperIReferenceState(
            amplitudes_real=tuple(
                _finite(value, label="reference real amplitude")
                for value in _sequence(reference.get("amplitudes_real"), label="reference real")
            ),
            amplitudes_imaginary=tuple(
                _finite(value, label="reference imaginary amplitude")
                for value in _sequence(reference.get("amplitudes_imaginary"), label="reference imaginary")
            ),
            qubit_count=_integer(reference.get("qubit_count"), label="reference qubits", minimum=1),
            source_label=str(reference.get("source_label", "")),
            state_fingerprint=str(reference.get("state_fingerprint", "")),
        ),
        checkpoint_sha256=str(prefix.get("checkpoint_sha256", "")),
        projective_state_fingerprint=str(prefix.get("projective_state_fingerprint", "")),
        problem_request_sha256=str(prefix.get("problem_request_sha256", "")),
        route_profile=str(prefix.get("route_profile", "")),
        route_contract_sha256=str(prefix.get("route_contract_sha256", "")),
        algorithmic_work=PaperIAlgorithmicWork(
            components=PaperIWorkComponents(**components),
            s_alg=work["s_alg"],
        ),
    )
    return compile_paper_i_prefix_qiskit_payload(typed)


def _normalize_compiled_cost(payload: Mapping[str, Any]) -> dict[str, Any]:
    convention = str(payload.get("compile_convention", ""))
    if convention != COMPILE_CONVENTION:
        raise Page8InputError("round-50 compiler convention drifted")

    def metric(*names: str) -> int:
        for name in names:
            value = payload.get(name)
            if value is not None:
                return _integer(value, label=f"compiled {name}")
        raise Page8InputError(f"compiled metric is absent: {names[0]}")

    w1q_raw = payload.get(
        "W1q", payload.get("qiskit_pretranspile_pauli_1q_work_total")
    )
    if w1q_raw is None:
        raise Page8InputError("compiled W1q is unavailable")
    return {
        "N2q": metric("N2q", "compiled_two_qubit_count", "compiled_count_2q_total"),
        "D2q": metric("D2q", "compiled_two_qubit_depth", "compiled_depth_2q_total"),
        "Dc": metric("Dc", "compiled_total_depth", "compiled_depth_total"),
        "W1q": _integer(w1q_raw, label="compiled W1q"),
        "B1q": (
            None
            if payload.get(
                "B1q", payload.get("qiskit_pretranspile_basis_change_1q_total")
            )
            is None
            else _integer(
                payload.get(
                    "B1q", payload.get("qiskit_pretranspile_basis_change_1q_total")
                ),
                label="compiled B1q",
            )
        ),
        "compile_convention": convention,
        "qiskit_basis_work_status": payload.get("qiskit_basis_work_status"),
        "qiskit_basis_work_schema": payload.get("qiskit_basis_work_schema"),
        "qiskit_version": payload.get("qiskit_version"),
        "generator_coefficients_sha256": payload.get(
            "generator_coefficients_sha256"
        ),
    }


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    initial_energy: float,
    compiler: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    execution_id = str(job["execution_id"])
    exact = _finite(job.get("exact_same_cutoff_energy"), label=f"{execution_id} exact energy")
    if summary.get("schema") != SUMMARY_SCHEMA or summary.get(
        "available_controller_rounds"
    ) != TARGET_ROUND:
        raise Page8InputError(f"{execution_id}: summary horizon drifted")
    provenance = _mapping(summary.get("provenance"), label=f"{execution_id} provenance")
    if (
        provenance.get("candidate_representation") != "single_pauli_word_v1"
        or provenance.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or provenance.get("route_profile") != ROUTE_PROFILE
        or provenance.get("qiskit_compile_convention") != COMPILE_CONVENTION
        or str(provenance.get("optimizer", "")).upper() != "POWELL"
        or provenance.get("optimizer_maxiter") != 200
        or provenance.get("seed") != 7
        or not math.isclose(
            _finite(provenance.get("exact_same_cutoff_energy"), label="summary exact"),
            exact,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        )
    ):
        raise Page8InputError(f"{execution_id}: summary provenance drifted")
    trace = _sequence(summary.get("accepted_error_trace"), label=f"{execution_id} trace")
    if len(trace) != TARGET_ROUND:
        raise Page8InputError(f"{execution_id}: trace is not 50 rounds")
    points = [{"k": 0, "error": abs(initial_energy - exact)}]
    for expected_round, raw in enumerate(trace, start=1):
        row = _mapping(raw, label=f"{execution_id} trace row")
        energy = _finite(row.get("accepted_energy"), label="accepted energy")
        error = _finite(row.get("absolute_energy_error"), label="accepted error", minimum=0.0)
        if row.get("controller_round") != expected_round or not math.isclose(
            error,
            abs(energy - exact),
            abs_tol=1.0e-12,
            rel_tol=1.0e-11,
        ):
            raise Page8InputError(f"{execution_id}: trace math drifted")
        points.append({"k": expected_round, "error": error})
    all_work = _work(summary.get("canonical_all_work"), label=f"{execution_id} work")
    requested = _sequence(summary.get("requested_rounds"), label="requested rounds")
    if len(requested) != 1:
        raise Page8InputError(f"{execution_id}: round-50 observation is not unique")
    row = _mapping(requested[0], label="round-50 observation")
    resources = _mapping(row.get("resources"), label="round-50 resources")
    prefix = _mapping(row.get("prefix"), label="round-50 prefix")
    if (
        row.get("status") != "available"
        or row.get("controller_round") != TARGET_ROUND
        or _work(row.get("algorithmic_work"), label="requested work") != all_work
        or _work(prefix.get("algorithmic_work"), label="prefix work") != all_work
        or prefix.get("controller_round") != TARGET_ROUND
        or prefix.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or prefix.get("route_profile") != ROUTE_PROFILE
        or resources.get("compile_convention") != COMPILE_CONVENTION
    ):
        raise Page8InputError(f"{execution_id}: round-50 observation drifted")
    serialized = {
        "N2q": _integer(resources.get("compiled_two_qubit_count"), label="serialized N2q"),
        "D2q": _integer(resources.get("compiled_two_qubit_depth"), label="serialized D2q"),
        "Dc": _integer(resources.get("compiled_total_depth"), label="serialized Dc"),
    }
    try:
        compiled_payload = dict(compiler(prefix))
        compiled = _normalize_compiled_cost(compiled_payload)
        if any(compiled[key] != serialized[key] for key in serialized):
            raise Page8InputError(
                f"{execution_id}: serialized and recompiled Qiskit costs disagree"
            )
        compile_failure = None
    except Page8InputError:
        raise
    except Exception as exc:  # Qiskit observation failure is non-scientific.
        compiled_payload = {}
        compiled = {
            **serialized,
            "W1q": None,
            "B1q": None,
            "compile_convention": COMPILE_CONVENTION,
            "qiskit_basis_work_status": "retryable_observation_failure",
            "qiskit_basis_work_schema": None,
            "qiskit_version": None,
            "generator_coefficients_sha256": None,
        }
        compile_failure = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "retryable": True,
        }
    plateau = _mapping(summary.get("effective_plateau"), label="effective plateau")
    marker_round = _integer(
        plateau.get("controller_round"), label="effective plateau round", minimum=1
    )
    marker_error = _finite(
        plateau.get("absolute_energy_error"), label="effective plateau error", minimum=0.0
    )
    if (
        plateau.get("policy") != "paper_i_effective_plateau_v1"
        or marker_round > TARGET_ROUND
        or not math.isclose(
            marker_error,
            float(points[marker_round]["error"]),
            abs_tol=1.0e-12,
            rel_tol=1.0e-11,
        )
    ):
        raise Page8InputError(f"{execution_id}: effective plateau drifted")
    return {
        "points": points,
        "marker": {
            "k": marker_round,
            "error": marker_error,
            "policy": "first_effective_plateau_prefix",
        },
        "terminal": {
            "k": TARGET_ROUND,
            "error": float(points[-1]["error"]),
            **compiled,
            "S_alg": all_work["s_alg"],
            "status": (
                "complete"
                if compile_failure is None
                else "science_complete_qiskit_basis_work_retryable"
            ),
        },
        "compile_failure": compile_failure,
        "compile_payload_sha256": (
            None
            if not compiled_payload
            else hashlib.sha256(canonical_json_bytes(compiled_payload)).hexdigest()
        ),
        "exact_same_cutoff_energy": exact,
    }


def validate_attempt_archive(
    path: Path,
    *,
    regime: str,
    package: Mapping[str, Any] | None = None,
    compiler: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    package_authority = (
        _load_package_authority(PACKAGE_DIR) if package is None else dict(package)
    )
    execution_id = expected_execution_id(regime)
    loaded = _read_attempt_archive(path.resolve())
    members = _mapping(loaded["members"], label="archive members")
    payloads = _mapping(loaded["payloads"], label="archive payloads")
    required_payloads = {
        "worker_attempt_receipt.json",
        "authority/job.json",
        "authority/execution_authorization.json",
        "authority/activation_manifest.json",
        "worker_outputs/worker_receipt.json",
        "worker_outputs/artifacts/execution_manifest.json",
        "worker_outputs/artifacts/paper_i_summary.json",
    }
    if set(payloads) != required_payloads:
        raise Page8InputError(f"{execution_id}: selected member closure drifted")
    attempt = payloads["worker_attempt_receipt.json"]
    job = payloads["authority/job.json"]
    authorization = payloads["authority/execution_authorization.json"]
    activation = payloads["authority/activation_manifest.json"]
    worker = payloads["worker_outputs/worker_receipt.json"]
    execution = payloads["worker_outputs/artifacts/execution_manifest.json"]
    summary = payloads["worker_outputs/artifacts/paper_i_summary.json"]
    attempt_digest = verify_self_digest(attempt, label=f"{execution_id} attempt receipt")
    job_digest = verify_self_digest(job, label=f"{execution_id} job")
    authorization_digest = verify_self_digest(
        authorization, label=f"{execution_id} authorization"
    )
    activation_digest = verify_self_digest(
        activation, label=f"{execution_id} activation"
    )
    worker_digest = verify_self_digest(worker, label=f"{execution_id} worker receipt")
    execution_digest = verify_self_digest(
        execution, label=f"{execution_id} execution manifest"
    )

    worker_rows = [
        dict(_mapping(row, label="attempt worker member"))
        for row in _sequence(attempt.get("worker_files"), label="attempt worker files")
    ]
    worker_names = [str(row.get("path", "")) for row in worker_rows]
    if len(worker_names) != len(set(worker_names)):
        raise Page8InputError(f"{execution_id}: worker member paths collide")
    expected_members = {
        *(f"worker_outputs/{name}" for name in worker_names),
        "authority/job.json",
        "authority/execution_authorization.json",
        "authority/activation_manifest.json",
        "worker_attempt_receipt.json",
    }
    if set(members) != expected_members:
        raise Page8InputError(f"{execution_id}: exact archive member closure drifted")
    for row in worker_rows:
        name = f"worker_outputs/{row['path']}"
        if not _binding_matches(_mapping(members[name], label=name), row):
            raise Page8InputError(f"{execution_id}: worker member binding drifted: {name}")
    external = {
        "authority/job.json": "job_file_sha256",
        "authority/execution_authorization.json": "authorization_file_sha256",
        "authority/activation_manifest.json": "activation_manifest_file_sha256",
    }
    for name, field in external.items():
        if _mapping(members[name], label=name).get("sha256") != attempt.get(field):
            raise Page8InputError(f"{execution_id}: {name} receipt binding drifted")

    package_jobs = _mapping(package_authority["jobs"], label="package jobs")
    package_protocols = _mapping(package_authority["protocols"], label="package protocols")
    package_job = _mapping(package_jobs[execution_id], label="package job binding")
    package_protocol = _mapping(
        package_protocols[execution_id], label="package protocol binding"
    )
    job_member = _mapping(members["authority/job.json"], label="job member")
    if (
        job.get("schema") != JOB_SCHEMA
        or job.get("sha256") != job_digest
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("execution_id") != execution_id
        or job.get("regime_id") != regime
        or job.get("nph") != NPH_BY_REGIME[regime]
        or job.get("target_horizon") != TARGET_ROUND
        or job.get("candidate_representation") != "single_pauli_word_v1"
        or job.get("insertion_policy") != "plateau_commutation"
        or job.get("active_gradient_policy") != "stationary_source_response_v1"
        or job.get("resource_weighting_scope") != "late_resource_weighting_v1"
        or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or job.get("plateau_prior_mean_decrease_ratio_threshold") != PLATEAU_RATIO
        or job.get("plateau_threshold_comparison") != PLATEAU_COMPARISON
        or job.get("plateau_trigger_source") != PLATEAU_TRIGGER
        or package_job.get("canonical_sha256") != job_digest
        or not _binding_matches(job_member, package_job)
        or package_protocol.get("canonical_sha256") != job.get("protocol_sha256")
        or package_protocol.get("sha256") != job.get("protocol_file_sha256")
    ):
        raise Page8InputError(f"{execution_id}: job/package authority drifted")

    activation_rows = {
        str(_mapping(row, label="activation execution").get("execution_id")): dict(row)
        for row in _sequence(activation.get("executions"), label="activation executions")
    }
    authorization_rows = {
        str(_mapping(row, label="activation authorization").get("execution_id")): dict(row)
        for row in _sequence(
            activation.get("execution_authorizations"),
            label="activation authorizations",
        )
    }
    sealed = _mapping(activation.get("sealed_package"), label="sealed package")
    remote_image = _mapping(activation.get("remote_image"), label="remote image")
    if (
        activation.get("sha256") != activation_digest
        or activation.get("package_id") != PACKAGE_ID
        or activation.get("campaign_id") != CAMPAIGN_ID
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not True
        or execution_id not in activation_rows
        or execution_id not in authorization_rows
        or _mapping(activation_rows[execution_id].get("job"), label="activation job").get(
            "canonical_sha256"
        )
        != job_digest
        or authorization_rows[execution_id].get("canonical_sha256")
        != authorization_digest
        or _mapping(sealed.get("manifest"), label="sealed manifest").get(
            "canonical_sha256"
        )
        != package_authority["manifest"]["sha256"]
        or _mapping(sealed.get("source_archive"), label="sealed source").get("sha256")
        != package_authority["source_archive"]["sha256"]
    ):
        raise Page8InputError(f"{execution_id}: activation authority drifted")
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("sha256") != authorization_digest
        or authorization.get("status") != "passed"
        or authorization.get("package_id") != PACKAGE_ID
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("execution_id") != execution_id
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not True
        or authorization.get("job_spec_sha256") != job_digest
        or authorization.get("protocol_sha256") != job.get("protocol_sha256")
        or authorization.get("package_manifest_sha256")
        != package_authority["manifest"]["sha256"]
        or authorization.get("source_archive_sha256")
        != package_authority["source_archive"]["sha256"]
        or authorization.get("remote_image_sha256") != remote_image.get("sha256")
        or attempt.get("source_archive_sha256")
        != authorization.get("source_archive_sha256")
        or attempt.get("image_sha256") != authorization.get("remote_image_sha256")
    ):
        raise Page8InputError(f"{execution_id}: execution authorization drifted")
    if (
        attempt.get("schema") != ATTEMPT_SCHEMA
        or attempt.get("sha256") != attempt_digest
        or attempt.get("execution_id") != execution_id
        or attempt.get("worker_exit_status") != 0
        or attempt.get("science_evidence_state") != "success_payload_closed_v2"
        or _integer(attempt.get("cluster_id"), label="cluster id") < 1
        or _integer(attempt.get("proc_id"), label="proc id") < 0
        or _integer(attempt.get("attempt_ordinal"), label="attempt ordinal", minimum=1)
        < 1
    ):
        raise Page8InputError(f"{execution_id}: scheduler attempt receipt drifted")

    artifact_members = {
        name.removeprefix("worker_outputs/artifacts/"): binding
        for name, binding in members.items()
        if name.startswith("worker_outputs/artifacts/")
    }
    required_artifacts = {
        "checkpoint.json",
        "estimator_ledger.json",
        "execution_manifest.json",
        "paper_i_summary.json",
        "result.json",
    }
    dynamic = set(artifact_members) - required_artifacts
    if (
        not required_artifacts.issubset(artifact_members)
        or len({name for name in dynamic if _ESTIMATOR_SIDECAR.fullmatch(name)}) != 1
        or len({name for name in dynamic if _RESUME_SIDECAR.fullmatch(name)}) != 1
        or any(
            not (_ESTIMATOR_SIDECAR.fullmatch(name) or _RESUME_SIDECAR.fullmatch(name))
            for name in dynamic
        )
    ):
        raise Page8InputError(f"{execution_id}: science artifact closure drifted")
    expected_worker_artifacts = [
        {
            "path": name,
            "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
        }
        for name, binding in sorted(artifact_members.items())
    ]
    expected_output_payloads = {
        name: {
            "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
        }
        for name, binding in sorted(artifact_members.items())
        if name != "execution_manifest.json"
    }
    if (
        worker.get("schema") != WORKER_RECEIPT_SCHEMA
        or worker.get("sha256") != worker_digest
        or worker.get("status") != "passed"
        or worker.get("execution_id") != execution_id
        or worker.get("job_spec_sha256") != job_digest
        or worker.get("authorization_sha256") != authorization_digest
        or worker.get("execution_manifest_sha256") != execution_digest
        or worker.get("controller_rounds_completed") != TARGET_ROUND
        or worker.get("artifacts") != expected_worker_artifacts
        or execution.get("schema") != EXECUTION_MANIFEST_SCHEMA
        or execution.get("sha256") != execution_digest
        or execution.get("status") != "passed"
        or execution.get("execution_id") != execution_id
        or execution.get("job_spec_sha256") != job_digest
        or execution.get("authorization_sha256") != authorization_digest
        or execution.get("protocol_sha256") != job.get("protocol_sha256")
        or execution.get("target_horizon") != TARGET_ROUND
        or execution.get("controller_rounds_completed") != TARGET_ROUND
        or execution.get("fresh_start") is not True
        or execution.get("source_checkpoint_consumed") is not False
        or execution.get("output_payloads") != expected_output_payloads
    ):
        raise Page8InputError(f"{execution_id}: worker/execution closure drifted")

    summary_projection = _validate_summary(
        summary,
        job=job,
        initial_energy=float(loaded["initial_energy"]),
        compiler=_compile_prefix_mapping if compiler is None else compiler,
    )
    source_bindings = {
        "archive": copy.deepcopy(loaded["archive"]),
        "package_manifest": copy.deepcopy(package_authority["manifest_binding"]),
        "job": {
            **copy.deepcopy(dict(job_member)),
            "canonical_sha256": job_digest,
        },
        "protocol": copy.deepcopy(dict(package_protocol)),
        "authorization": {
            **copy.deepcopy(dict(members["authority/execution_authorization.json"])),
            "canonical_sha256": authorization_digest,
        },
        "activation_manifest": {
            **copy.deepcopy(dict(members["authority/activation_manifest.json"])),
            "canonical_sha256": activation_digest,
            "activation_id": activation.get("activation_id"),
        },
        "worker_attempt_receipt": {
            **copy.deepcopy(dict(members["worker_attempt_receipt.json"])),
            "canonical_sha256": attempt_digest,
        },
        "worker_receipt": {
            **copy.deepcopy(dict(members["worker_outputs/worker_receipt.json"])),
            "canonical_sha256": worker_digest,
        },
        "execution_manifest": {
            **copy.deepcopy(
                dict(members["worker_outputs/artifacts/execution_manifest.json"])
            ),
            "canonical_sha256": execution_digest,
        },
        "result": copy.deepcopy(dict(artifact_members["result.json"])),
        "summary": copy.deepcopy(dict(artifact_members["paper_i_summary.json"])),
        "checkpoint": copy.deepcopy(dict(artifact_members["checkpoint.json"])),
        "estimator_ledger": copy.deepcopy(
            dict(artifact_members["estimator_ledger.json"])
        ),
    }
    return {
        "regime_id": regime,
        "regime_label": REGIME_LABELS[regime],
        "nph": NPH_BY_REGIME[regime],
        "execution_id": execution_id,
        "cluster_id": attempt["cluster_id"],
        "proc_id": attempt["proc_id"],
        "attempt_ordinal": attempt["attempt_ordinal"],
        **summary_projection,
        "source_bindings": source_bindings,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise Page8InputError(f"stale JSON temporary exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode(
                    "utf-8"
                )
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_adapter(
    attempts: Mapping[str, Path],
    *,
    output: Path,
    package_dir: Path = PACKAGE_DIR,
    compiler: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if set(attempts) != set(REGIME_ORDER):
        raise Page8InputError("exactly one attempt is required for every regime")
    resolved_paths = [attempts[regime].resolve() for regime in REGIME_ORDER]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise Page8InputError("attempt archive paths must be unique")
    package = _load_package_authority(package_dir.resolve())
    cells = [
        validate_attempt_archive(
            attempts[regime],
            regime=regime,
            package=package,
            compiler=compiler,
        )
        for regime in REGIME_ORDER
    ]
    adapter = digested(
        {
            "schema": BASE_ADAPTER_SCHEMA,
            "status": "passed_six_completed_cells",
            "classification": "supplemental_candidate_diagnostic_not_adopted_evidence",
            "paper_evidence_adopted": False,
            "page_id": BASE_PAGE_ID,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "package_manifest": copy.deepcopy(package["manifest_binding"]),
            "regime_order": list(REGIME_ORDER),
            "completed_regimes": list(REGIME_ORDER),
            "candidate_representation": "single_pauli_word_v1",
            "active_gradient_policy": "stationary_source_response_v1",
            "resource_weighting_scope": "late_resource_weighting_v1",
            "insertion_policy": "plateau_commutation",
            "phase3_population_activation": (
                "same_round_authenticated_insertion_plateau_domain_open_v1"
            ),
            "plateau_prior_mean_decrease_ratio_threshold": PLATEAU_RATIO,
            "plateau_threshold_comparison": PLATEAU_COMPARISON,
            "plateau_trigger_source": PLATEAU_TRIGGER,
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
            "route_profile": ROUTE_PROFILE,
            "target_controller_rounds": TARGET_ROUND,
            "error_metric": "same_cutoff_absolute_energy_error",
            "cost_tuple": ["N2q", "D2q", "Dc", "W1q", "S_alg"],
            "cost_round": TARGET_ROUND,
            "compile_convention": COMPILE_CONVENTION,
            "cells": cells,
        }
    )
    _atomic_write_json(output, adapter)
    return adapter


def _validate_base_adapter(path: Path) -> dict[str, Any]:
    adapter = _load_json_file(path, label="page-8 base adapter")
    verify_self_digest(adapter, label="page-8 base adapter")
    cells = _sequence(adapter.get("cells"), label="page-8 base cells")
    if (
        adapter.get("schema") != BASE_ADAPTER_SCHEMA
        or adapter.get("status") != "passed_six_completed_cells"
        or adapter.get("paper_evidence_adopted") is not False
        or adapter.get("page_id") != BASE_PAGE_ID
        or adapter.get("package_id") != PACKAGE_ID
        or tuple(adapter.get("regime_order", ())) != REGIME_ORDER
        or len(cells) != len(REGIME_ORDER)
        or tuple(str(_mapping(row, label="adapter cell").get("regime_id")) for row in cells)
        != REGIME_ORDER
    ):
        raise Page8InputError("page-8 base adapter identity drifted")
    return adapter


def _append_comparator_projection(path: Path) -> dict[str, Any]:
    adapter = _load_json_file(path, label="Append-ADAPT comparator adapter")
    canonical_sha256 = verify_self_digest(
        adapter, label="Append-ADAPT comparator adapter"
    )
    raw_cells = _sequence(
        adapter.get("cells"), label="Append-ADAPT comparator cells"
    )
    if (
        adapter.get("schema") != APPEND_ADAPTER_SCHEMA
        or adapter.get("status") != "passed"
        or adapter.get("package_id") != APPEND_PACKAGE_ID
        or tuple(adapter.get("regime_order", ())) != REGIME_ORDER
        or tuple(adapter.get("completed_regimes", ())) != REGIME_ORDER
        or tuple(adapter.get("pending_regimes", ())) != ()
        or len(raw_cells) != len(REGIME_ORDER)
    ):
        raise Page8InputError("Append-ADAPT comparator identity drifted")
    by_regime: dict[str, dict[str, Any]] = {}
    cost_fields = ("N2q", "D2q", "Dc", "W1q", "S_alg")
    for raw_cell in raw_cells:
        cell = _mapping(raw_cell, label="Append-ADAPT comparator cell")
        regime = str(cell.get("regime_id"))
        if regime not in REGIME_ORDER or regime in by_regime:
            raise Page8InputError("Append-ADAPT comparator regime closure drifted")
        if cell.get("nph") != NPH_BY_REGIME[regime]:
            raise Page8InputError(f"{regime}: Append-ADAPT cutoff drifted")
        raw_points = _sequence(
            cell.get("points"), label=f"{regime} Append-ADAPT points"
        )
        rounds: list[int] = []
        points: list[dict[str, Any]] = []
        for index, raw_point in enumerate(raw_points):
            point = _mapping(
                raw_point, label=f"{regime} Append-ADAPT point {index}"
            )
            round_index = _integer(
                point.get("round"),
                label=f"{regime} Append-ADAPT point round",
            )
            rounds.append(round_index)
            error = _finite(
                point.get("delta_e"),
                label=f"{regime} Append-ADAPT point error",
            )
            if error < 0.0:
                raise Page8InputError(
                    f"{regime}: Append-ADAPT error is negative"
                )
            points.append({"k": round_index, "error": error})
        if rounds != list(range(APPEND_TRAJECTORY_ROUND + 1)) or [
            row["k"] for row in points
        ] != list(range(APPEND_TRAJECTORY_ROUND + 1)):
            raise Page8InputError(
                f"{regime}: Append-ADAPT points are not exact rounds 0..70"
            )
        endpoints = _mapping(
            cell.get("endpoints"), label=f"{regime} Append-ADAPT endpoints"
        )
        endpoint = _mapping(
            endpoints.get("round_50"),
            label=f"{regime} Append-ADAPT round-50 endpoint",
        )
        endpoint_error = _finite(
            endpoint.get("delta_e"),
            label=f"{regime} Append-ADAPT round-50 error",
        )
        if endpoint.get("round") != TARGET_ROUND or not math.isclose(
            endpoint_error,
            float(points[TARGET_ROUND]["error"]),
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ):
            raise Page8InputError(
                f"{regime}: Append-ADAPT round-50 endpoint drifted"
            )
        costs = _mapping(
            endpoint.get("costs"), label=f"{regime} Append-ADAPT costs"
        )
        if set(costs) != set(cost_fields):
            raise Page8InputError(f"{regime}: Append-ADAPT cost tuple drifted")
        projected_costs: dict[str, int] = {}
        for field in cost_fields:
            projected_costs[field] = _integer(
                costs.get(field),
                label=f"{regime} Append-ADAPT {field}",
            )
        compile_receipt = _mapping(
            endpoint.get("compile"), label=f"{regime} Append-ADAPT compile"
        )
        if compile_receipt.get("compile_convention") != COMPILE_CONVENTION:
            raise Page8InputError(
                f"{regime}: Append-ADAPT compile convention drifted"
            )
        trajectory_endpoint = _mapping(
            endpoints.get("round_70"),
            label=f"{regime} Append-ADAPT round-70 endpoint",
        )
        trajectory_error = _finite(
            trajectory_endpoint.get("delta_e"),
            label=f"{regime} Append-ADAPT round-70 error",
        )
        if (
            trajectory_endpoint.get("round") != APPEND_TRAJECTORY_ROUND
            or not math.isclose(
                trajectory_error,
                float(points[APPEND_TRAJECTORY_ROUND]["error"]),
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
        ):
            raise Page8InputError(
                f"{regime}: Append-ADAPT round-70 endpoint drifted"
            )
        by_regime[regime] = {
            "execution_id": str(cell.get("execution_id")),
            "exact_same_cutoff_energy": _finite(
                cell.get("exact_same_cutoff_energy"),
                label=f"{regime} Append-ADAPT exact reference",
            ),
            "points": points,
            "marker": {
                "k": TARGET_ROUND,
                "error": endpoint_error,
                "policy": "terminal_common_horizon",
            },
            "terminal": {
                "k": TARGET_ROUND,
                "error": endpoint_error,
                **projected_costs,
                "compile_convention": COMPILE_CONVENTION,
            },
            "trajectory_terminal": {
                "k": APPEND_TRAJECTORY_ROUND,
                "error": trajectory_error,
            },
            "source": copy.deepcopy(cell.get("source")),
        }
    if tuple(by_regime) != REGIME_ORDER:
        raise Page8InputError("Append-ADAPT comparator ordering drifted")
    return {
        "binding": {
            **file_binding(path),
            "canonical_sha256": canonical_sha256,
        },
        "package_id": APPEND_PACKAGE_ID,
        "source_classification": adapter.get("classification"),
        "source_authentication_summary": copy.deepcopy(
            adapter.get("source_authentication_summary")
        ),
        "limitations": copy.deepcopy(adapter.get("limitations", [])),
        "cells": by_regime,
    }


def attach_append_comparator(
    *, ra_adapter_path: Path, append_adapter_path: Path, output: Path
) -> dict[str, Any]:
    base = _validate_base_adapter(ra_adapter_path)
    append = _append_comparator_projection(append_adapter_path)
    cells: list[dict[str, Any]] = []
    for raw_cell in base["cells"]:
        cell = copy.deepcopy(dict(_mapping(raw_cell, label="RA adapter cell")))
        regime = str(cell["regime_id"])
        comparator = append["cells"][regime]
        if not math.isclose(
            float(cell["exact_same_cutoff_energy"]),
            float(comparator["exact_same_cutoff_energy"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise Page8InputError(
                f"{regime}: RA/Append same-cutoff reference drifted"
            )
        cell["append_adapt"] = comparator
        cells.append(cell)
    comparison = copy.deepcopy(base)
    comparison.pop("sha256", None)
    comparison.update(
        {
            "schema": ADAPTER_SCHEMA,
            "status": "passed_six_completed_cells_with_append_comparator",
            "page_id": PAGE_ID,
            "comparison_method": "Append-ADAPT",
            "comparison_round": TARGET_ROUND,
            "comparison_horizon_policy": (
                "append_trajectory_round_70_with_ra_and_cost_anchor_round_50_v1"
            ),
            "append_adapter": {
                **append["binding"],
                "package_id": append["package_id"],
                "classification": append["source_classification"],
                "source_authentication_summary": append[
                    "source_authentication_summary"
                ],
                "limitations": append["limitations"],
            },
            "cells": cells,
        }
    )
    comparison = digested(comparison)
    _atomic_write_json(output, comparison)
    return comparison


def validate_adapter(path: Path) -> dict[str, Any]:
    adapter = _load_json_file(path, label="page-8 comparison adapter")
    verify_self_digest(adapter, label="page-8 comparison adapter")
    cells = _sequence(adapter.get("cells"), label="page-8 comparison cells")
    if (
        adapter.get("schema") != ADAPTER_SCHEMA
        or adapter.get("status")
        != "passed_six_completed_cells_with_append_comparator"
        or adapter.get("paper_evidence_adopted") is not False
        or adapter.get("page_id") != PAGE_ID
        or adapter.get("package_id") != PACKAGE_ID
        or adapter.get("comparison_method") != "Append-ADAPT"
        or adapter.get("comparison_round") != TARGET_ROUND
        or tuple(adapter.get("regime_order", ())) != REGIME_ORDER
        or len(cells) != len(REGIME_ORDER)
        or tuple(
            str(_mapping(row, label="adapter cell").get("regime_id"))
            for row in cells
        )
        != REGIME_ORDER
        or any(
            not isinstance(_mapping(row, label="adapter cell").get("append_adapt"), Mapping)
            for row in cells
        )
    ):
        raise Page8InputError("page-8 comparison adapter identity drifted")
    return adapter


def _format_sci(value: float) -> str:
    return f"{float(value):.2e}"


def _format_s_alg(value: int) -> str:
    if value == 0:
        return "0.0e0"
    exponent = int(math.floor(math.log10(value)))
    coefficient = value / (10**exponent)
    if round(coefficient, 1) >= 10.0:
        coefficient /= 10.0
        exponent += 1
    return f"{coefficient:.1f}e{exponent}"


def _format_cost(value: Mapping[str, Any]) -> str:
    fields = ("N2q", "D2q", "Dc", "W1q", "S_alg")
    return "(" + ", ".join(
        "pending"
        if value.get(field) is None
        else _format_s_alg(int(value[field]))
        if field == "S_alg"
        else str(value[field])
        for field in fields
    ) + ")"


def render_plot(adapter: Mapping[str, Any], *, png_path: Path, pdf_path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    mpl.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "font.family": "serif",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.1, 4.1), constrained_layout=True)
    for ax, raw_cell in zip(axes.flat, adapter["cells"], strict=True):
        cell = _mapping(raw_cell, label="plot cell")
        ra_points = [_mapping(row, label="RA plot point") for row in cell["points"]]
        append = _mapping(cell["append_adapt"], label="Append-ADAPT plot cell")
        append_points = [
            _mapping(row, label="Append-ADAPT plot point")
            for row in append["points"]
        ]
        ra_x = [int(row["k"]) for row in ra_points]
        ra_y = [max(float(row["error"]), PLOT_FLOOR) for row in ra_points]
        append_x = [int(row["k"]) for row in append_points]
        append_y = [
            max(float(row["error"]), PLOT_FLOOR) for row in append_points
        ]
        ax.plot(append_x, append_y, color="#4C78A8", linewidth=1.6)
        ax.plot(ra_x, ra_y, color="#009E73", linewidth=1.8)
        append_marker = _mapping(
            append["marker"], label="Append-ADAPT plot marker"
        )
        ax.scatter(
            [int(append_marker["k"])],
            [max(float(append_marker["error"]), PLOT_FLOOR)],
            color="#4C78A8",
            marker="o",
            s=30,
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        marker = _mapping(cell["marker"], label="plot marker")
        ax.scatter(
            [int(marker["k"])],
            [max(float(marker["error"]), PLOT_FLOOR)],
            color="#009E73",
            marker="P",
            s=44,
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        ax.set_yscale("log")
        ax.set_xlim(0, APPEND_TRAJECTORY_ROUND)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.grid(True, which="major", alpha=0.22, linewidth=0.55)
        ax.set_title(f"{cell['regime_label']} ($n_{{ph}}={cell['nph']}$)")
        index = REGIME_ORDER.index(str(cell["regime_id"]))
        if index // 3 == 1:
            ax.set_xlabel("ADAPT controller round")
        if index % 3 == 0:
            ax.set_ylabel(r"same-cutoff $|\Delta E|$")
    legend = [
        Line2D(
            [0],
            [0],
            color="#4C78A8",
            linewidth=1.6,
            marker="o",
            markersize=5,
            label="Append-ADAPT trajectory to k=70 (cost marker k=50)",
        ),
        Line2D(
            [0],
            [0],
            color="#009E73",
            linewidth=1.8,
            marker="P",
            markersize=6,
            label="RA singleton: Phase III activates on insertion plateau",
        )
    ]
    fig.suptitle(
        "Singleton comparison: Append trajectory to k=70; RA and costs at k=50",
        fontsize=11.2,
        fontweight="bold",
    )
    fig.legend(
        handles=legend,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        title="Markers: Append cost anchor k=50; RA first effective-plateau prefix",
    )
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    result = value
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result


def write_page_tex(adapter: Mapping[str, Any], *, plot_pdf: Path, tex_path: Path) -> None:
    rows = []
    for raw in adapter["cells"]:
        cell = _mapping(raw, label="table cell")
        terminal = _mapping(cell["terminal"], label="RA table terminal")
        append = _mapping(cell["append_adapt"], label="Append-ADAPT table cell")
        append_terminal = _mapping(
            append["terminal"], label="Append-ADAPT table terminal"
        )
        rows.append(
            f"{_latex_escape(str(cell['regime_label']))} & {cell['marker']['k']} & "
            f"{_format_sci(float(terminal['error']))} & "
            f"{_latex_escape(_format_cost(terminal))} & "
            f"{_format_sci(float(append_terminal['error']))} & "
            f"{_latex_escape(_format_cost(append_terminal))} \\\\"
        )
    tex = rf"""
\documentclass[10pt,letterpaper]{{article}}
\usepackage[landscape,margin=0.16in]{{geometry}}
\usepackage{{amsmath,booktabs,graphicx}}
\usepackage[T1]{{fontenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
\includegraphics[width=0.92\textwidth,height=3.75in,keepaspectratio]{{{plot_pdf.as_posix()}}}
\vspace{{0.15em}}

\tiny
\setlength{{\tabcolsep}}{{2.4pt}}
\resizebox{{0.985\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
Regime & $k_{{pl}}^{{\rm RA}}$ & $|\Delta E_{{50}}^{{\rm RA}}|$ &
$C_{{50}}^{{\rm RA}}$ & $|\Delta E_{{50}}^{{\rm Append}}|$ &
$C_{{50}}^{{\rm Append}}$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\end{{center}}
\vspace{{-0.25em}}
\tiny
$C=(N_{{2q}},D_{{2q}},D_c,W_{{1q}},S_{{\rm alg}})$ at controller round 50;
all errors use exact diagonalization at the identical phonon cutoff.  Both
methods use the common source-locked Table-I compiler (optimization level 0,
seed 7, reference state included).  Phase III remains unpopulated until the
authenticated same-round commutation-reduced insertion-plateau predicate opens
at the strict marginal-to-prior-mean decrease-ratio threshold $10^{{-4}}$.
The page is diagnostic and is not adopted Paper-I evidence.
\end{{document}}
""".strip()
    tex_path.write_text(tex + "\n", encoding="utf-8")


def _compile_page(tex_path: Path, page_pdf: Path) -> None:
    output_dir = tex_path.parent
    latexmk = shutil.which("latexmk")
    tectonic = shutil.which("tectonic")
    if latexmk:
        command = [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={output_dir}",
            str(tex_path),
        ]
    elif tectonic:
        command = [
            tectonic,
            "--keep-logs",
            "--reruns",
            "2",
            "--outdir",
            str(output_dir),
            str(tex_path),
        ]
    else:
        raise Page8InputError("neither latexmk nor tectonic is available")
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not page_pdf.is_file():
        raise Page8InputError("page-8 LaTeX build did not emit its PDF")
    from pypdf import PdfReader

    if len(PdfReader(str(page_pdf), strict=False).pages) != 1:
        raise Page8InputError("page-8 asset is not exactly one page")


def build_assets(
    adapter: Mapping[str, Any], *, asset_dir: Path, asset_stem: str
) -> dict[str, Path]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", asset_stem):
        raise Page8InputError("asset stem is unsafe")
    asset_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "plot_png": asset_dir / f"{asset_stem}_plot.png",
        "plot_pdf": asset_dir / f"{asset_stem}_plot.pdf",
        "page_tex": asset_dir / f"{asset_stem}.tex",
        "page_pdf": asset_dir / f"{asset_stem}.pdf",
    }
    render_plot(adapter, png_path=assets["plot_png"], pdf_path=assets["plot_pdf"])
    write_page_tex(adapter, plot_pdf=assets["plot_pdf"], tex_path=assets["page_tex"])
    _compile_page(assets["page_tex"], assets["page_pdf"])
    return assets


def _page_content_hashes(path: Path) -> list[str]:
    from pypdf import PdfReader

    hashes = []
    for page in PdfReader(str(path), strict=False).pages:
        contents = page.get_contents()
        payload = b"" if contents is None else contents.get_data()
        hashes.append(hashlib.sha256(payload).hexdigest())
    return hashes


def append_page8(
    *,
    target_pdf: Path,
    target_provenance: Path,
    adapter_path: Path,
    assets: Mapping[str, Path],
) -> dict[str, Any]:
    adapter = validate_adapter(adapter_path)
    if (
        not target_pdf.is_file()
        or target_pdf.is_symlink()
        or not target_provenance.is_file()
        or target_provenance.is_symlink()
    ):
        raise Page8InputError("target PDF/provenance is unavailable or unsafe")
    provenance = _load_json_file(target_provenance, label="target provenance")
    layout = _mapping(provenance.get("layout"), label="target layout")
    outputs = _mapping(provenance.get("outputs"), label="target outputs")
    current_binding = _mapping(
        outputs.get("partial_progress_pdf"), label="target PDF binding"
    )
    observed_binding = file_binding(target_pdf)
    if not _binding_matches(observed_binding, current_binding):
        raise Page8InputError("target PDF/provenance byte binding drifted")
    before_hashes = _page_content_hashes(target_pdf)
    if (
        len(before_hashes) != 7
        or layout.get("page_count") != 7
        or layout.get("page_7") is None
        or layout.get("page_8") is not None
        or provenance.get(REPORT_KEY) is not None
    ):
        raise Page8InputError("target is not the supported seven-page report")
    required_assets = {"plot_png", "plot_pdf", "page_tex", "page_pdf"}
    if set(assets) != required_assets:
        raise Page8InputError("page-8 asset closure drifted")
    for role, path in assets.items():
        if not path.is_file() or path.is_symlink():
            raise Page8InputError(f"page-8 {role} is unavailable or unsafe")
    from pypdf import PdfReader, PdfWriter

    page_reader = PdfReader(str(assets["page_pdf"]), strict=False)
    if len(page_reader.pages) != 1:
        raise Page8InputError("page-8 asset is not exactly one page")
    temporary_pdf = target_pdf.with_name(f".{target_pdf.name}.page8.tmp")
    temporary_provenance = target_provenance.with_name(
        f".{target_provenance.name}.page8.tmp"
    )
    rollback_pdf = target_pdf.with_name(f".{target_pdf.name}.page8.rollback")
    for path in (temporary_pdf, temporary_provenance, rollback_pdf):
        if path.exists() or path.is_symlink():
            raise Page8InputError(f"stale page-8 temporary exists: {path}")
    writer = PdfWriter()
    for page in PdfReader(str(target_pdf), strict=False).pages:
        writer.add_page(page)
    writer.add_page(page_reader.pages[0])
    try:
        with temporary_pdf.open("xb") as stream:
            writer.write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        after_hashes = _page_content_hashes(temporary_pdf)
        if len(after_hashes) != 8 or after_hashes[:7] != before_hashes:
            raise Page8InputError("page-8 append altered a preserved page")
        combined_binding = file_binding(temporary_pdf)
        combined_binding["path"] = str(target_pdf.resolve())
        asset_bindings = {role: file_binding(path) for role, path in assets.items()}
        adapter_binding = {
            **file_binding(adapter_path),
            "canonical_sha256": adapter["sha256"],
        }
        report = {
            "schema": PAGE_REPORT_SCHEMA,
            "page_id": PAGE_ID,
            "classification": adapter["classification"],
            "paper_evidence_adopted": False,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "adapter": adapter_binding,
            "comparison_method": adapter["comparison_method"],
            "comparison_round": adapter["comparison_round"],
            "comparison_horizon_policy": adapter["comparison_horizon_policy"],
            "append_adapter": copy.deepcopy(adapter["append_adapter"]),
            "route_profile": adapter["route_profile"],
            "route_contract_sha256": adapter["route_contract_sha256"],
            "error_metric": adapter["error_metric"],
            "cost_tuple": copy.deepcopy(adapter["cost_tuple"]),
            "cost_round": TARGET_ROUND,
            "cells": [
                {
                    "regime_id": cell["regime_id"],
                    "nph": cell["nph"],
                    "execution_id": cell["execution_id"],
                    "cluster_id": cell["cluster_id"],
                    "proc_id": cell["proc_id"],
                    "attempt_ordinal": cell["attempt_ordinal"],
                    "points": copy.deepcopy(cell["points"]),
                    "marker": copy.deepcopy(cell["marker"]),
                    "terminal": copy.deepcopy(cell["terminal"]),
                    "append_adapt": copy.deepcopy(cell["append_adapt"]),
                    "compile_failure": copy.deepcopy(cell["compile_failure"]),
                    "source_bindings": copy.deepcopy(cell["source_bindings"]),
                }
                for cell in adapter["cells"]
            ],
            "outputs": asset_bindings,
            "structural_validation": {
                "pages_before": 7,
                "pages_after": 8,
                "preserved_page_content_sha256": before_hashes,
                "new_page_8_content_sha256": after_hashes[7],
            },
        }
        updated = copy.deepcopy(provenance)
        updated["layout"]["page_count"] = 8
        updated["layout"]["page_8"] = PAGE_ID
        updated[REPORT_KEY] = report
        updated["outputs"]["partial_progress_pdf"] = combined_binding
        updated["outputs"].update(
            {
                "phase3_on_plateau_singleton_page8_adapter": adapter_binding,
                **{
                    f"phase3_on_plateau_singleton_page8_{role}": binding
                    for role, binding in asset_bindings.items()
                },
            }
        )
        with temporary_provenance.open("xb") as stream:
            stream.write(
                json.dumps(updated, indent=2, sort_keys=True, allow_nan=False).encode(
                    "utf-8"
                )
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(target_pdf, rollback_pdf)
        os.replace(temporary_pdf, target_pdf)
        try:
            os.replace(temporary_provenance, target_provenance)
        except Exception:
            os.replace(rollback_pdf, target_pdf)
            raise
        rollback_pdf.unlink(missing_ok=True)
    except Exception:
        temporary_pdf.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)
        rollback_pdf.unlink(missing_ok=True)
        raise
    return {
        "status": "appended_page_8",
        "output_pdf": str(target_pdf),
        "output_provenance": str(target_provenance),
        "pages": 8,
        "preserved_pages": 7,
        "completed_regimes": list(REGIME_ORDER),
        "pdf_sha256": sha256_file(target_pdf),
    }


def _regime_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        regime, separator, path_text = raw.partition("=")
        if not separator or regime not in REGIME_ORDER or not path_text:
            raise Page8InputError("--attempt must be REGIME=/path/to/archive.tar.gz")
        if regime in result:
            raise Page8InputError(f"duplicate attempt regime: {regime}")
        result[regime] = Path(path_text).expanduser().resolve()
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--attempt",
        action="append",
        default=[],
        metavar="REGIME=ARCHIVE",
        help="Repeat exactly once for each of the six regimes.",
    )
    result.add_argument("--adapter", type=Path, required=True)
    result.add_argument(
        "--append-adapter",
        type=Path,
        default=DEFAULT_APPEND_ADAPTER,
        help="Authenticated six-regime Append-ADAPT R70 adapter; round 50 is used.",
    )
    result.add_argument("--target-pdf", type=Path, required=True)
    result.add_argument("--target-provenance", type=Path, required=True)
    result.add_argument("--asset-dir", type=Path, required=True)
    result.add_argument("--asset-stem", required=True)
    result.add_argument("--package-dir", type=Path, default=PACKAGE_DIR)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        attempts = _regime_paths(args.attempt)
        build_adapter(
            attempts,
            output=args.adapter.resolve(),
            package_dir=args.package_dir.resolve(),
        )
        adapter = attach_append_comparator(
            ra_adapter_path=args.adapter.resolve(),
            append_adapter_path=args.append_adapter.resolve(),
            output=args.adapter.resolve(),
        )
        assets = build_assets(
            adapter,
            asset_dir=args.asset_dir.resolve(),
            asset_stem=args.asset_stem,
        )
        result = append_page8(
            target_pdf=args.target_pdf.resolve(),
            target_provenance=args.target_provenance.resolve(),
            adapter_path=args.adapter.resolve(),
            assets=assets,
        )
    except (OSError, Page8InputError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

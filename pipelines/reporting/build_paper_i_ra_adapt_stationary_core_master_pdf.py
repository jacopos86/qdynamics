#!/usr/bin/env python3
"""Build the Paper-I stationary RA-ADAPT 48-cell master report.

Final mode consumes the package's existing explicit 48-attempt selection.
Preview mode is intentionally data-free and emits a visibly non-evidentiary
two-page layout while the CHTC results are pending.
Partial-progress mode consumes only unambiguous passed attempts from a
validated subset, fills the remaining cells as visibly pending, and cannot
emit the canonical final-report name or claim paper evidence.
Cross-revision progress can append explicitly separated diagnostic pages
without admitting those sources into the validated 48-cell matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

from pipelines.reporting.paper_i_qiskit_cost_tuple import (
    PAPER_I_QISKIT_COST_TUPLE_FIELDS,
    PAPER_I_QISKIT_COST_TUPLE_LATEX,
    paper_i_cost_tuple_latex,
    qiskit_cost_fields,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "stationary_core_full48_r50_20260728_v6_chtc"
)
PACKAGE_DIR = DEFAULT_PACKAGE_DIR
PACKAGE_ID = (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_v6_chtc"
)
PACKAGE_RELATIVE_ROOT = PACKAGE_DIR.relative_to(REPO_ROOT).as_posix()
CORE_MATERIALIZATION_ID = "ra_adapt_stationary_late_core_v10"
CORE_MATERIALIZATION_PATTERN = re.compile(
    r"ra_adapt_stationary_late_core_v[0-9]+"
)
OUTPUT_DIR = REPO_ROOT / (
    "output/pdf/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_v6"
)
STEM = "paper_i_ra_adapt_stationary_core_full48_r50_20260728_v6"
CROSS_REVISION_OUTPUT_DIR = REPO_ROOT / (
    "output/pdf/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
)
CROSS_REVISION_STEM = (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_"
    "evolving_partial_progress"
)
SELECTION_SCHEMA = "paper_i_ra_adapt_stationary_core_attempt_selection_v1"
VALIDATION_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_fetched_validation_v1"
)
RECOVERY_ADAPTER_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_recovery_adapter_v1"
)
RECOVERY_CROSS_CAMPAIGN_CLASS = (
    "cross_campaign_science_equivalent_passed_attempt_v1"
)
RECOVERY_G5_UNEXERCISED_CLASS = (
    "completed_science_g5_plateau_domain_unexercised_v1"
)
LOCAL_PAUSED_ALWAYS_PACKAGE_ID = (
    "paper_i_ra_adapt_always_factorial48_r50_20260730_v1_chtc"
)
GLOBAL_SINGLETON_WW_DIAGNOSTIC_SCHEMA = (
    "paper_i_ra_adapt_global_singleton_weak_weak_comparison_diagnostic_v1"
)
GLOBAL_SINGLETON_WW_CAMPAIGN_ID = (
    "paper_i_ra_adapt_global_singleton_insertion_comparison_v1"
)
GLOBAL_SINGLETON_WW_CROSS_ARM_SHA256 = (
    "59f8b94a4ff2f2070765ddbc1fac56ba2c69c1f55a1d40b86bfca460c3d0b2e6"
)
GLOBAL_SINGLETON_WW_POLICIES = (
    "append_commutation_reduced",
    "plateau_commutation",
)

REGIME_ORDER = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
REGIME_LABELS = {
    "weak_weak": ("WW", "Weak--weak"),
    "intermediate_weak": ("IW", "Intermediate--weak"),
    "strong_weak_u8": ("SW", "Strong--weak"),
    "weak_strong": ("WS", "Weak--strong"),
    "intermediate_strong": ("IS", "Intermediate--strong"),
    "strong_strong_u8": ("SS", "Strong--strong"),
}
REPRESENTATIONS = {
    "macro": "macro_generator_v1",
    "singleton": "single_pauli_word_v1",
}
REPRESENTATION_TITLES = {
    "macro": "Macro-generator stationary-source core",
    "singleton": "Single-Pauli-word stationary-source core",
}
METHODS: Mapping[str, Mapping[str, Any]] = {
    "always": {
        "label": "Stationary RA - always insertion",
        "short": "RA-always",
        "color": "#8B1A1A",
        "marker": "*",
        "linewidth": 2.05,
    },
    "plateau": {
        "label": "Stationary RA - plateau insertion",
        "short": "RA-plateau",
        "color": "#E45756",
        "marker": "D",
        "linewidth": 1.8,
    },
    "no_insertion": {
        "label": "Stationary RA - no insertion",
        "short": "RA-none",
        "color": "#F2A0A0",
        "marker": "s",
        "linewidth": 1.45,
    },
    "append": {
        "label": "Conventional unwhitened ADAPT",
        "short": "ADAPT",
        "color": "#4C78A8",
        "marker": "o",
        "linewidth": 1.55,
    },
}
METHOD_ORDER = ("always", "plateau", "no_insertion", "append")
PLOT_TUPLE_MARKERS = {
    "always": r"\star",
    "plateau": r"\diamond",
    "no_insertion": r"\boxminus",
    "append": r"\bullet",
}
QISKIT_PLATEAU_MACRO_ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_plateau_insertion_"
    "qiskit_transpile_cost_v1"
)
QISKIT_PLATEAU_MACRO_EXECUTION_ID = (
    "qiskit_cost_pilot__strong_weak_u8__nph3__ra_macro_plateau"
)
QISKIT_PLATEAU_MACRO_ROUTE_PROFILE = (
    "paper_i_ra_adapt__macro_generator_v1__"
    "insertion_commutation_plateau_v1__"
    "stationary_source_response_v1__"
    "all_phase_resource_weighting_v1__"
    "qiskit_full_ansatz_transpile_cost_all_phases_v1"
)
QISKIT_ALWAYS_MACRO_ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_always_insertion_qiskit_transpile_cost_v1"
)
QISKIT_ALWAYS_MACRO_EXECUTION_ID = (
    "qiskit_cost_always13__strong_weak_u8__nph3__ra_macro_always"
)
QISKIT_ALWAYS_MACRO_ROUTE_PROFILE = (
    "paper_i_ra_adapt__macro_generator_v1__"
    "full_commutation_reduced__"
    "stationary_source_response_v1__"
    "all_phase_resource_weighting_v1__"
    "qiskit_full_ansatz_transpile_cost_all_phases_v1"
)
FIXED_COMPARISON_ROUND = 10
FIXED_COMPARISON_EXECUTION_IDS = frozenset(
    {
        "core__strong_weak_u8__nph3__append_macro",
        "core__strong_weak_u8__nph3__ra_macro_append_only",
        "core__strong_weak_u8__nph3__ra_macro_plateau",
    }
)
MATCHED_SINGLETON_ROUND = 33
MATCHED_SINGLETON_EXECUTION_IDS = frozenset(
    {
        "core__strong_strong_u8__nph7__append_singleton",
        "core__strong_strong_u8__nph7__ra_singleton_plateau",
    }
)
QISKIT_GLOBAL_SINGLETON_ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_plateau_commutation_"
    "qiskit_transpile_cost_v1"
)
QISKIT_GLOBAL_SINGLETON_EXECUTION_ID = (
    "qiskit_cost_pilot__strong_strong_u8__nph7__"
    "ra_global_singleton_plateau_commutation"
)
QISKIT_GLOBAL_SINGLETON_ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v1__"
    "global_guarded_singleton_phase_i__identity_phase_ii__"
    "stationary_source_response_v1__all_phase_resource_weighting_v1__"
    "qiskit_full_ansatz_transpile_cost_all_phases_v1"
)
CUMULATIVE_PLATEAU_MACRO_EXECUTION_ID = (
    "core__intermediate_strong__nph7__ra_macro_plateau"
)
CUMULATIVE_PLATEAU_MACRO_STYLE = {
    "label": (
        "Stationary RA - cumulative-relative plateau "
        r"($10^{-4}$, diagnostic k=20)"
    ),
    "color": "#009E73",
    "marker": "P",
    "linewidth": 2.2,
}


class ReportInputError(ValueError):
    """Raised when selected runtime evidence cannot support the final report."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportInputError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportInputError(f"{label} must be a JSON object")
    return payload


def _without_none(value: Any) -> Any:
    """Project typed JSON onto fields that carry serialized data."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_none(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, (list, tuple)):
        return [_without_none(item) for item in value]
    return value


def _configure_package_dir(package_dir: Path) -> str:
    global CORE_MATERIALIZATION_ID
    global OUTPUT_DIR
    global PACKAGE_DIR
    global PACKAGE_ID
    global PACKAGE_RELATIVE_ROOT
    global STEM

    resolved = package_dir.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ReportInputError(
            "stationary-core package must be inside the active repository"
        ) from exc
    manifest = _load_object(
        resolved / "package_manifest.json",
        label="package manifest identity",
    )
    package_id = str(manifest.get("package_id", ""))
    if re.fullmatch(
        (
            r"paper_i_ra_adapt_stationary_core_full48_r50_"
            r"20260728_v[0-9]+_chtc"
        ),
        package_id,
    ) is None:
        raise ReportInputError(
            "package manifest is not a stationary-core successor"
        )
    core = _mapping(
        manifest.get("core_final_receipt"),
        label="package core final receipt",
    )
    core_path = PurePosixPath(str(core.get("path", "")))
    materialization_ids = [
        part
        for part in core_path.parts
        if CORE_MATERIALIZATION_PATTERN.fullmatch(part)
    ]
    if (
        len(materialization_ids) != 1
        or core_path.name != "final_publication_receipt.json"
    ):
        raise ReportInputError(
            "package does not bind one stationary-core science authority"
        )
    CORE_MATERIALIZATION_ID = materialization_ids[0]
    PACKAGE_DIR = resolved
    PACKAGE_RELATIVE_ROOT = relative.as_posix()
    PACKAGE_ID = package_id
    STEM = package_id.removesuffix("_chtc")
    OUTPUT_DIR = REPO_ROOT / "output/pdf" / STEM
    return package_id


def _package_contract() -> Any:
    module_name = (
        "_paper_i_stationary_core_package_contract_"
        + hashlib.sha256(str(PACKAGE_DIR).encode("utf-8")).hexdigest()[:12]
    )
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    path = PACKAGE_DIR / "package_contract.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ReportInputError("cannot load the selected package contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _verified_object(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    payload = _load_object(path, label=label)
    try:
        digest = _package_contract().verify_self_digest(payload, label=label)
    except Exception as exc:
        raise ReportInputError(f"{label} self-digest failed: {exc}") from exc
    return payload, str(digest)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _verify_generic_self_digest(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> str:
    unsigned = dict(payload)
    digest = unsigned.pop("sha256", None)
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or _canonical_sha256(unsigned) != digest
    ):
        raise ReportInputError(f"{label} self-digest failed")
    return digest


def _expected_jobs() -> dict[str, dict[str, Any]]:
    _configure_package_dir(PACKAGE_DIR)
    jobs: dict[str, dict[str, Any]] = {}
    for path in sorted((PACKAGE_DIR / "jobs").glob("*.json")):
        payload, _ = _verified_object(path, label=f"package job {path.name}")
        execution_id = str(payload.get("execution_id", ""))
        if not execution_id or path.name != f"{execution_id}.json":
            raise ReportInputError(f"package job identity drifted: {path}")
        jobs[execution_id] = payload
    if len(jobs) != 48:
        raise ReportInputError(
            f"expected exactly 48 package jobs, observed {len(jobs)}"
        )
    matrix = {
        (
            str(row.get("regime_id")),
            str(row.get("candidate_representation")),
            _method_key(str(row.get("route_id"))),
        )
        for row in jobs.values()
    }
    expected = {
        (regime, representation, method)
        for regime in REGIME_ORDER
        for representation in REPRESENTATIONS.values()
        for method in METHOD_ORDER
    }
    if matrix != expected:
        raise ReportInputError("package jobs do not form the exact 6x2x4 matrix")
    return jobs


def _method_key(route_id: str) -> str:
    if route_id in {"append_macro", "append_singleton"}:
        return "append"
    if route_id.endswith("_append_only"):
        return "no_insertion"
    if route_id.endswith("_plateau"):
        return "plateau"
    if route_id.endswith("_always"):
        return "always"
    raise ReportInputError(f"unknown stationary-core route: {route_id}")


def _safe_attempt_path(fetched_dir: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or "." in pure.parts or ".." in pure.parts:
        raise ReportInputError(f"unsafe selected attempt path: {relative}")
    path = fetched_dir.joinpath(*pure.parts)
    if not path.is_file() or path.is_symlink():
        raise ReportInputError(f"selected attempt is unavailable: {relative}")
    return path


def _json_member(
    archive: tarfile.TarFile,
    name: str,
    *,
    label: str,
) -> tuple[dict[str, Any], str, int]:
    try:
        member = archive.getmember(name)
    except KeyError as exc:
        raise ReportInputError(f"{label} is missing from selected archive") from exc
    if not member.isfile():
        raise ReportInputError(f"{label} is not a regular archive member")
    stream = archive.extractfile(member)
    if stream is None:
        raise ReportInputError(f"{label} has no readable bytes")
    raw = stream.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportInputError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ReportInputError(f"{label} must be a JSON object")
    return payload, _sha256_bytes(raw), len(raw)


def _attempt_member_names(execution_id: str) -> dict[str, str]:
    """Return the exact paths emitted by the selected worker attempt wrapper."""

    return {
        "job": (
            f"{PACKAGE_RELATIVE_ROOT}/jobs/{execution_id}.json"
        ),
        "worker": "worker_outputs/worker_receipt.json",
        "manifest": "worker_outputs/execution_manifest.json",
        "result": "worker_outputs/result.json",
        "summary": "worker_outputs/summary.json",
    }


def _load_attempt_payloads(
    attempt: Path,
    *,
    execution_id: str,
) -> dict[str, tuple[dict[str, Any], str, int]]:
    names = _attempt_member_names(execution_id)
    try:
        archive = tarfile.open(attempt, "r:gz")
    except tarfile.TarError as exc:
        raise ReportInputError(
            f"{execution_id}: selected archive is unreadable"
        ) from exc
    with archive:
        return {
            role: _json_member(
                archive,
                name,
                label=f"{execution_id} {role}",
            )
            for role, name in names.items()
        }


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportInputError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ReportInputError(f"{label} must be finite")
    return result


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ReportInputError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ReportInputError(f"{label} must be an integer") from exc
    if result < minimum or result != value:
        raise ReportInputError(f"{label} must be an integer >= {minimum}")
    return result


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportInputError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ReportInputError(f"{label} must be a sequence")
    return value


def _closure_context(
    *,
    execution_id: str,
    job: Mapping[str, Any],
    worker: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], float]:
    if (
        job.get("execution_id") != execution_id
        or job.get("package_id") != PACKAGE_ID
        or worker.get("execution_id") != execution_id
        or worker.get("status") != "passed"
        or manifest.get("execution_id") != execution_id
        or manifest.get("status") != "passed"
        or manifest.get("paper_facing_result_allowed") is not True
        or manifest.get("maximum_controller_rounds_override") is not None
    ):
        raise ReportInputError(f"{execution_id}: final execution closure failed")
    closure = _mapping(
        worker.get("scientific_closure"),
        label=f"{execution_id} scientific closure",
    )
    expected_gates = [f"G{index}" for index in range(1, 14)]
    gates = _mapping(closure.get("gates"), label=f"{execution_id} gates")
    if (
        closure.get("status") != "passed"
        or closure.get("full_controller_rounds") != 50
        or closure.get("gate_ids") != expected_gates
        or set(gates) != set(expected_gates)
        or any(
            not isinstance(gates[gate], Mapping)
            or gates[gate].get("status") != "passed"
            for gate in expected_gates
        )
    ):
        raise ReportInputError(f"{execution_id}: G1-G13 closure failed")
    g2 = _mapping(
        _mapping(gates["G2"], label=f"{execution_id} G2").get("evidence"),
        label=f"{execution_id} G2 evidence",
    )
    exact = _mapping(
        g2.get("verified_ed_reference"),
        label=f"{execution_id} same-cutoff reference",
    )
    if exact.get("status") != "passed":
        raise ReportInputError(f"{execution_id}: same-cutoff reference failed")
    return closure, _finite(exact.get("E_ED"), label=f"{execution_id} E_ED")


def _artifact_binding_map(
    worker: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    rows = _sequence(worker.get("artifact_bindings"), label="artifact bindings")
    out: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        row = _mapping(raw, label="artifact binding")
        role = str(row.get("role", ""))
        if not role or role in out:
            raise ReportInputError("artifact role is empty or duplicated")
        out[role] = row
    return out


def _verify_loaded_artifact(
    bindings: Mapping[str, Mapping[str, Any]],
    role: str,
    file_sha256: str,
    size_bytes: int,
) -> None:
    row = _mapping(bindings.get(role), label=f"{role} artifact binding")
    if (
        row.get("sha256") != file_sha256
        or row.get("size_bytes") != size_bytes
    ):
        raise ReportInputError(f"{role} artifact binding disagrees with bytes")


def _ra_prefix(
    result: Mapping[str, Any],
    *,
    controller_round: int,
    expected_s_alg: int | None = None,
) -> Any:
    """Reconstruct one authenticated RA prefix at an accepted round."""

    from pipelines.reporting.paper_i_run_summary import (
        PaperIAlgorithmicWork,
        PaperIPrefixCompileInput,
        PaperIPrefixOperator,
        PaperIPrefixPauliTerm,
        PaperIReferenceState,
        PaperIWorkComponents,
    )

    run = _mapping(result.get("run"), label="RA run")
    trajectory = _sequence(
        run.get("accepted_trajectory"), label="RA accepted trajectory"
    )
    replay = _sequence(run.get("scientific_replay"), label="RA scientific replay")
    work_rows = _sequence(
        _mapping(
            run.get("canonical_reporting"), label="RA canonical reporting"
        ).get("accepted_prefix_work"),
        label="RA accepted-prefix work",
    )
    available_rounds = len(trajectory)
    if (
        available_rounds < 1
        or len(replay) != available_rounds
        or len(work_rows) != available_rounds
    ):
        raise ReportInputError("RA prefix requires nonempty aligned rows")
    round_index = _integer(
        controller_round,
        label="RA prefix controller round",
        minimum=1,
    )
    if round_index > available_rounds:
        raise ReportInputError(
            "RA prefix controller round exceeds the available trajectory"
        )
    zero_index = round_index - 1
    state = _mapping(trajectory[zero_index], label="RA selected state")
    replay_row = _mapping(replay[zero_index], label="RA selected replay")
    if replay_row.get("accepted_state") != state:
        raise ReportInputError("RA selected replay state drifted")
    checkpoint = _mapping(
        replay_row.get("checkpoint"), label="RA selected checkpoint"
    )
    reference_raw = _mapping(
        _mapping(
            run.get("canonical_reporting"), label="RA canonical reporting"
        ).get("reference_state"),
        label="RA reference state",
    )
    reference = PaperIReferenceState(
        amplitudes_real=tuple(
            _finite(value, label="RA reference real amplitude")
            for value in _sequence(
                reference_raw.get("amplitudes_real"),
                label="RA reference real amplitudes",
            )
        ),
        amplitudes_imaginary=tuple(
            _finite(value, label="RA reference imaginary amplitude")
            for value in _sequence(
                reference_raw.get("amplitudes_imaginary"),
                label="RA reference imaginary amplitudes",
            )
        ),
        qubit_count=_integer(
            reference_raw.get("qubit_count"),
            label="RA reference qubit count",
            minimum=1,
        ),
        source_label=str(reference_raw.get("source_label", "")),
        state_fingerprint=str(reference_raw.get("state_fingerprint", "")),
    )
    labels = tuple(
        str(value)
        for value in _sequence(
            checkpoint.get("ordered_operator_labels"),
            label="RA selected ordered labels",
        )
    )
    state_labels = tuple(
        str(value)
        for value in _sequence(state.get("operators"), label="RA state operators")
    )
    depth = _integer(
        checkpoint.get("active_ansatz_depth"),
        label="RA selected active depth",
        minimum=1,
    )
    if labels != state_labels or len(labels) != depth:
        raise ReportInputError("RA selected operator order/depth drifted")
    blocks = _sequence(
        checkpoint.get("parameter_blocks"), label="RA selected parameter blocks"
    )
    operators = []
    for index, raw_block in enumerate(blocks):
        block = _mapping(raw_block, label=f"RA parameter block {index}")
        terms = tuple(
            PaperIPrefixPauliTerm(
                pauli_exyz=str(term["pauli_exyz"]),
                coefficient_real=_finite(
                    term.get("coefficient_real"),
                    label="RA Pauli coefficient real",
                ),
                coefficient_imaginary=_finite(
                    term.get("coefficient_imaginary"),
                    label="RA Pauli coefficient imaginary",
                ),
                qubit_count=_integer(
                    term.get("qubit_count"),
                    label="RA Pauli term qubit count",
                    minimum=1,
                ),
            )
            for term in (
                _mapping(value, label="RA runtime term")
                for value in _sequence(
                    block.get("runtime_terms"), label="RA runtime terms"
                )
            )
        )
        operators.append(
            PaperIPrefixOperator(
                candidate_label=str(block.get("candidate_label", "")),
                logical_index=_integer(
                    block.get("logical_index"),
                    label="RA logical index",
                ),
                runtime_start=_integer(
                    block.get("runtime_start"),
                    label="RA runtime start",
                ),
                runtime_count=_integer(
                    block.get("runtime_count"),
                    label="RA runtime count",
                    minimum=1,
                ),
                execution_mode=str(block.get("execution_mode", "")),
                runtime_terms=terms,
            )
        )
    work_raw = _mapping(
        work_rows[zero_index], label="RA selected prefix work"
    )
    components_raw = _mapping(
        work_raw.get("components"), label="RA terminal work components"
    )
    components = PaperIWorkComponents(
        n_h_outer=_integer(
            components_raw.get("n_h_outer"), label="RA N_H_outer"
        ),
        n_h_refit=_integer(
            components_raw.get("n_h_refit"), label="RA N_H_refit"
        ),
        n_grad=_integer(components_raw.get("n_grad"), label="RA N_grad"),
        n_metric=_integer(
            components_raw.get("n_metric"), label="RA N_metric"
        ),
    )
    prefix_s_alg = _integer(
        work_raw.get("s_alg"), label="RA selected prefix S_alg"
    )
    if (
        components.s_alg != prefix_s_alg
        or (
            expected_s_alg is not None
            and prefix_s_alg != expected_s_alg
        )
    ):
        raise ReportInputError("RA selected prefix S_alg drifted")
    work = PaperIAlgorithmicWork(
        components=components,
        s_alg=prefix_s_alg,
    )
    route = _mapping(run.get("route"), label="RA route")
    problem = _mapping(run.get("problem"), label="RA problem")
    if (
        checkpoint.get("outer_iteration") != round_index
        or state.get("controller_round") != round_index
        or checkpoint.get("active_ansatz_depth") != depth
        or checkpoint.get("strict_replay_passed") is not True
        or not math.isclose(
            _finite(
                checkpoint.get("strict_replay_fidelity"),
                label="RA terminal replay fidelity",
            ),
            1.0,
            abs_tol=1.0e-10,
            rel_tol=0.0,
        )
        or checkpoint.get("estimator_ledger_status") != "complete"
        or checkpoint.get("estimator_ledger_s_alg") != prefix_s_alg
        or checkpoint.get("route_profile") != route.get("profile")
        or checkpoint.get("route_contract_sha256")
        != route.get("contract_sha256")
        or checkpoint.get("projective_state_fingerprint")
        != state.get("projective_state_fingerprint")
    ):
        raise ReportInputError("RA selected checkpoint authentication failed")
    return PaperIPrefixCompileInput(
        source_method="ra_adapt",
        controller_round=round_index,
        active_ansatz_depth=depth,
        ordered_operator_labels=labels,
        operators=tuple(operators),
        logical_parameters=tuple(
            _finite(value, label="RA selected logical parameter")
            for value in _sequence(
                checkpoint.get("logical_parameters"),
                label="RA selected logical parameters",
            )
        ),
        runtime_parameters=tuple(
            _finite(value, label="RA selected runtime parameter")
            for value in _sequence(
                checkpoint.get("runtime_parameters"),
                label="RA selected runtime parameters",
            )
        ),
        reference_state=reference,
        checkpoint_sha256=str(checkpoint.get("checkpoint_sha256", "")),
        projective_state_fingerprint=str(
            state.get("projective_state_fingerprint", "")
        ),
        problem_request_sha256=str(
            problem.get("problem_request_sha256", "")
        ),
        route_profile=str(route.get("profile", "")),
        route_contract_sha256=str(route.get("contract_sha256", "")),
        algorithmic_work=work,
    )


def _ra_terminal_prefix(
    result: Mapping[str, Any],
    *,
    s_alg: int,
) -> Any:
    """Backward-compatible round-50 RA prefix reconstruction."""

    return _ra_prefix(
        result,
        controller_round=50,
        expected_s_alg=s_alg,
    )


def _compile_prefix_qiskit(
    prefix: Any,
    *,
    compiler: Callable[[Any], Any] | None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if int(getattr(prefix, "controller_round", -1)) < 1:
        raise ReportInputError(
            "Paper-I Qiskit compilation requires an accepted prefix"
        )
    if compiler is None:
        from pipelines.reporting.paper_i_run_summary import (
            compile_paper_i_prefix_qiskit_payload,
        )

        compiled = compile_paper_i_prefix_qiskit_payload(prefix)
    else:
        compiled = compiler(prefix)
    if isinstance(compiled, Mapping):
        convention = str(compiled.get("compile_convention", ""))
        payload: Mapping[str, Any] = compiled
    else:
        convention = str(getattr(compiled, "compile_convention", ""))
        payload = {
            "N2q": getattr(compiled, "compiled_two_qubit_count", None),
            "D2q": getattr(compiled, "compiled_two_qubit_depth", None),
            "Dc": getattr(compiled, "compiled_total_depth", None),
            "W1q": getattr(
                compiled,
                "qiskit_pretranspile_pauli_1q_work_total",
                getattr(compiled, "W1q", None),
            ),
            "B1q": getattr(
                compiled,
                "qiskit_pretranspile_basis_change_1q_total",
                getattr(compiled, "B1q", None),
            ),
            "qiskit_basis_work_status": getattr(
                compiled, "qiskit_basis_work_status", None
            ),
            "qiskit_basis_work_schema": getattr(
                compiled, "qiskit_basis_work_schema", None
            ),
        }
    if convention != "table_i_basis_gate_transpile_v1":
        raise ReportInputError("prefix Qiskit compiler convention drifted")
    try:
        normalized = qiskit_cost_fields(
            {
                **dict(payload),
                "metrics": {
                    "N2q": payload.get(
                        "N2q",
                        payload.get(
                            "compiled_two_qubit_count",
                            payload.get("compiled_count_2q_total"),
                        ),
                    ),
                    "D2q": payload.get(
                        "D2q",
                        payload.get(
                            "compiled_two_qubit_depth",
                            payload.get("compiled_depth_2q_total"),
                        ),
                    ),
                    "Dc": payload.get(
                        "Dc",
                        payload.get(
                            "compiled_total_depth",
                            payload.get("compiled_depth_total"),
                        ),
                    ),
                    "W1q": payload.get("W1q"),
                    "B1q": payload.get("B1q"),
                    "qiskit_basis_work_status": payload.get(
                        "qiskit_basis_work_status"
                    ),
                    "qiskit_basis_work_schema": payload.get(
                        "qiskit_basis_work_schema"
                    ),
                },
            }
        )
    except (TypeError, ValueError) as exc:
        raise ReportInputError(
            f"prefix five-coordinate Qiskit cost is unavailable: {exc}"
        ) from exc
    values = {
        "N2q": normalized["N2q"],
        "D2q": normalized["D2q"],
        "Dc": normalized["Dc"],
        "W1q": normalized["W1q"],
        "B1q": normalized["B1q"],
        "qiskit_basis_work_status": normalized[
            "qiskit_basis_work_status"
        ],
        "qiskit_basis_work_schema": normalized[
            "qiskit_basis_work_schema"
        ],
    }
    return values, str(prefix.checkpoint_sha256), dict(payload)


def _compile_terminal_qiskit(
    prefix: Any,
    *,
    compiler: Callable[[Any], Any] | None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Backward-compatible round-50 Qiskit compilation wrapper."""

    if int(getattr(prefix, "controller_round", -1)) != 50:
        raise ReportInputError(
            "Paper-I terminal Qiskit compilation requires controller round 50"
        )
    return _compile_prefix_qiskit(prefix, compiler=compiler)


def _fixed_prefix_qiskit_observation(
    prefix: Any,
    *,
    error: float,
    compiler: Callable[[Any], Any] | None,
) -> dict[str, Any]:
    """Compile one authenticated common-round prefix for route comparison."""

    resources, checkpoint_sha256, payload = _compile_prefix_qiskit(
        prefix,
        compiler=compiler,
    )
    return {
        "k": int(prefix.controller_round),
        "error": float(error),
        **resources,
        "S_alg": int(prefix.algorithmic_work.s_alg),
        "checkpoint_sha256": checkpoint_sha256,
        "compile_convention": "table_i_basis_gate_transpile_v1",
        "qiskit_version": payload.get("qiskit_version"),
        "status": "complete",
    }


def _curve_marker(
    points: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    plateau = summary.get("effective_plateau")
    if isinstance(plateau, Mapping):
        raw_round = plateau.get("controller_round")
        if (
            not isinstance(raw_round, bool)
            and isinstance(raw_round, int)
            and 0 <= raw_round <= 50
        ):
            for point in points:
                if point.get("k") == raw_round:
                    return {
                        "k": raw_round,
                        "error": float(point["error"]),
                        "policy": "first_effective_plateau_prefix",
                    }
    terminal = points[-1]
    return {
        "k": int(terminal["k"]),
        "error": float(terminal["error"]),
        "policy": "terminal_observed_point",
    }


def _qiskit_plateau_checkpoint_projection(
    checkpoint_path: Path,
    *,
    exact_energy: float,
    expected_rounds: int = 50,
) -> dict[str, Any]:
    """Stream the compact reporting projection from the large local checkpoint."""

    try:
        import ijson
    except ModuleNotFoundError as exc:
        raise ReportInputError(
            "the optional Qiskit-plateau diagnostic requires ijson to "
            "stream its large checkpoint"
        ) from exc

    scalar_events = {"string", "number", "boolean", "null"}
    root_prefixes = {
        "adapt_vqe.S_alg",
        "adapt_vqe.S_unique",
        "adapt_vqe.S_alg_components.N_H_outer",
        "adapt_vqe.S_alg_components.N_H_refit",
        "adapt_vqe.S_alg_components.N_grad",
        "adapt_vqe.S_alg_components.N_metric",
        "adapt_vqe.ansatz_depth",
        "adapt_vqe.final_full_refit.executed",
        "adapt_vqe.history_checkpoint_complete",
        "adapt_vqe.history_count",
        "adapt_vqe.history_tail_count",
        "adapt_vqe.logical_num_parameters",
        "adapt_vqe.nfev_total",
        "adapt_vqe.num_parameters",
        "adapt_vqe.partial_checkpoint",
        "adapt_vqe.stop_reason",
        "adapt_vqe.success",
        (
            "adapt_vqe.terminal_active_prefix_checkpoint."
            "active_ansatz_depth"
        ),
    }
    history_prefix = "adapt_vqe.history.item"
    history_fields = {
        "depth",
        "energy_after_opt",
        "energy_before_opt",
        "max_grad",
        "selected_op",
        "selected_position",
    }
    root_values: dict[str, Any] = {}
    history_rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    try:
        with checkpoint_path.open("rb") as stream:
            for prefix, event, value in ijson.parse(stream):
                if prefix in root_prefixes and event in scalar_events:
                    root_values[prefix] = value
                if prefix == history_prefix and event == "start_map":
                    current = {}
                elif (
                    current is not None
                    and prefix.startswith(history_prefix + ".")
                    and prefix.count(".") == 3
                    and event in scalar_events
                ):
                    field = prefix.rsplit(".", 1)[-1]
                    if field in history_fields:
                        current[field] = value
                elif (
                    prefix == history_prefix
                    and event == "end_map"
                    and current is not None
                ):
                    history_rows.append(current)
                    current = None
    except (OSError, ValueError) as exc:
        raise ReportInputError(
            f"Qiskit-plateau checkpoint stream failed: {exc}"
        ) from exc

    required_roots = root_prefixes - {"adapt_vqe.stop_reason"}
    missing_roots = sorted(required_roots - set(root_values))
    if missing_roots:
        raise ReportInputError(
            "Qiskit-plateau checkpoint lacks reporting fields: "
            + ", ".join(missing_roots)
        )
    if len(history_rows) != expected_rounds:
        raise ReportInputError(
            "Qiskit-plateau checkpoint history length drifted"
        )

    points: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    previous_after: float | None = None
    for expected_depth, raw in enumerate(history_rows, start=1):
        missing = sorted(history_fields - set(raw))
        if missing:
            raise ReportInputError(
                "Qiskit-plateau checkpoint history row lacks fields: "
                + ", ".join(missing)
            )
        depth = _integer(
            raw["depth"],
            label="Qiskit-plateau checkpoint depth",
            minimum=1,
        )
        before = _finite(
            raw["energy_before_opt"],
            label=f"Qiskit-plateau round {depth} energy before",
        )
        after = _finite(
            raw["energy_after_opt"],
            label=f"Qiskit-plateau round {depth} energy after",
        )
        position = _integer(
            raw["selected_position"],
            label=f"Qiskit-plateau round {depth} insertion position",
        )
        if depth != expected_depth or position >= depth:
            raise ReportInputError(
                "Qiskit-plateau checkpoint history identity drifted"
            )
        if previous_after is not None and not math.isclose(
            before,
            previous_after,
            abs_tol=1.0e-11,
            rel_tol=1.0e-11,
        ):
            raise ReportInputError(
                "Qiskit-plateau checkpoint energy continuity drifted"
            )
        if expected_depth == 1:
            points.append(
                {
                    "k": 0,
                    "energy": before,
                    "error": abs(before - exact_energy),
                }
            )
        points.append(
            {
                "k": depth,
                "energy": after,
                "error": abs(after - exact_energy),
            }
        )
        normalized_rows.append(
            {
                "depth": depth,
                "energy_before_opt": before,
                "energy_after_opt": after,
                "max_grad": _finite(
                    raw["max_grad"],
                    label=f"Qiskit-plateau round {depth} max gradient",
                ),
                "selected_op": str(raw["selected_op"]),
                "selected_position": position,
            }
        )
        previous_after = after

    components = {
        name: _integer(
            root_values[f"adapt_vqe.S_alg_components.{name}"],
            label=f"Qiskit-plateau {name}",
        )
        for name in ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
    }
    s_alg = _integer(
        root_values["adapt_vqe.S_alg"],
        label="Qiskit-plateau S_alg",
    )
    if sum(components.values()) != s_alg:
        raise ReportInputError(
            "Qiskit-plateau checkpoint S_alg components do not close"
        )
    if (
        root_values["adapt_vqe.history_checkpoint_complete"] is not True
        or root_values["adapt_vqe.partial_checkpoint"] is not True
        or root_values["adapt_vqe.success"] is not False
        or root_values["adapt_vqe.final_full_refit.executed"] is not False
        or root_values.get("adapt_vqe.stop_reason") is not None
        or _integer(
            root_values["adapt_vqe.history_count"],
            label="Qiskit-plateau history count",
        )
        != expected_rounds
        or _integer(
            root_values["adapt_vqe.history_tail_count"],
            label="Qiskit-plateau history tail count",
        )
        != expected_rounds
        or _integer(
            root_values["adapt_vqe.ansatz_depth"],
            label="Qiskit-plateau ansatz depth",
        )
        != expected_rounds
        or _integer(
            root_values[
                "adapt_vqe.terminal_active_prefix_checkpoint."
                "active_ansatz_depth"
            ],
            label="Qiskit-plateau terminal active depth",
        )
        != expected_rounds
    ):
        raise ReportInputError(
            "Qiskit-plateau checkpoint completion state drifted"
        )

    from pipelines.reporting.paper_i_run_summary import (
        PaperIErrorTracePoint,
        select_paper_i_effective_plateau,
    )

    plateau_selection = select_paper_i_effective_plateau(
        tuple(
            PaperIErrorTracePoint(
                controller_round=int(row["k"]),
                absolute_energy_error=float(row["error"]),
            )
            for row in points[1:]
        )
    )
    plateau = points[int(plateau_selection.controller_round)]
    interior_rows = [
        row
        for row in normalized_rows
        if int(row["selected_position"]) < int(row["depth"]) - 1
    ]
    return {
        "points": points,
        "marker": {
            "k": int(plateau["k"]),
            "error": float(plateau["error"]),
            "policy": str(plateau_selection.policy),
        },
        "terminal": {
            "k": expected_rounds,
            "energy": float(points[-1]["energy"]),
            "error": float(points[-1]["error"]),
            "S_alg": s_alg,
        },
        "accounting": {
            "S_alg": s_alg,
            "S_unique": _integer(
                root_values["adapt_vqe.S_unique"],
                label="Qiskit-plateau S_unique",
            ),
            "components": components,
            "nfev_total": _integer(
                root_values["adapt_vqe.nfev_total"],
                label="Qiskit-plateau nfev total",
            ),
        },
        "parameterization": {
            "logical_num_parameters": _integer(
                root_values["adapt_vqe.logical_num_parameters"],
                label="Qiskit-plateau logical parameter count",
            ),
            "runtime_num_parameters": _integer(
                root_values["adapt_vqe.num_parameters"],
                label="Qiskit-plateau runtime parameter count",
            ),
        },
        "insertion": {
            "first_interior_round": (
                None
                if not interior_rows
                else int(interior_rows[0]["depth"])
            ),
            "interior_count": len(interior_rows),
            "append_position_count": len(normalized_rows) - len(interior_rows),
        },
        "history_rows": normalized_rows,
        "checkpoint_state": {
            "history_checkpoint_complete": True,
            "partial_checkpoint": True,
            "success": False,
            "stop_reason": None,
            "final_full_refit_executed": False,
        },
    }


def _qiskit_plateau_prefix_from_checkpoint(
    checkpoint_path: Path,
    *,
    protocol_path: Path,
    controller_round: int,
) -> Any:
    """Reconstruct one authenticated prefix from the summary-failed run."""

    try:
        import ijson
    except ModuleNotFoundError as exc:
        raise ReportInputError(
            "the Qiskit-plateau prefix reconstruction requires ijson"
        ) from exc

    selected: Mapping[str, Any] | None = None
    try:
        with checkpoint_path.open("rb") as stream:
            for index, raw in enumerate(
                ijson.items(
                    stream,
                    "adapt_vqe.active_prefix_checkpoints.item",
                ),
                start=1,
            ):
                if index == controller_round:
                    selected = _mapping(
                        raw,
                        label="Qiskit-plateau selected prefix checkpoint",
                    )
                    break
    except (OSError, ValueError) as exc:
        raise ReportInputError(
            f"Qiskit-plateau prefix stream failed: {exc}"
        ) from exc
    if selected is None:
        raise ReportInputError(
            "Qiskit-plateau checkpoint lacks the requested prefix"
        )

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
    from pipelines.static_adapt.ra_adapt.contracts import (
        load_resolved_ra_adapt_protocol,
    )

    import numpy as np

    protocol = load_resolved_ra_adapt_protocol(protocol_path)
    problem = _append_problem_from_protocol(protocol)
    reference_array = np.asarray(
        problem.reference_state.build_state(),
        dtype=complex,
    ).reshape(-1)
    reference_array = reference_array / float(np.linalg.norm(reference_array))
    reference = PaperIReferenceState(
        amplitudes_real=tuple(float(value.real) for value in reference_array),
        amplitudes_imaginary=tuple(
            float(value.imag) for value in reference_array
        ),
        qubit_count=int(problem.layout.total_qubits),
        source_label=str(problem.reference_state.source_label),
        state_fingerprint=projective_state_fingerprint(reference_array),
    )

    round_index = _integer(
        selected.get("outer_iteration"),
        label="Qiskit-plateau selected outer iteration",
        minimum=1,
    )
    depth = _integer(
        selected.get("active_ansatz_depth"),
        label="Qiskit-plateau selected active depth",
        minimum=1,
    )
    labels = tuple(
        str(value)
        for value in _sequence(
            selected.get("ordered_active_operator_labels"),
            label="Qiskit-plateau selected labels",
        )
    )
    parameterization = _mapping(
        selected.get("parameterization"),
        label="Qiskit-plateau selected parameterization",
    )
    blocks = _sequence(
        parameterization.get("blocks"),
        label="Qiskit-plateau selected parameter blocks",
    )
    if (
        round_index != controller_round
        or depth != controller_round
        or len(labels) != depth
        or len(blocks) != depth
    ):
        raise ReportInputError(
            "Qiskit-plateau selected prefix identity drifted"
        )
    operators: list[Any] = []
    expected_runtime_start = 0
    for logical_index, raw in enumerate(blocks):
        block = _mapping(
            raw,
            label=f"Qiskit-plateau parameter block {logical_index}",
        )
        runtime_terms = tuple(
            PaperIPrefixPauliTerm(
                pauli_exyz=str(term.get("pauli_exyz", "")),
                coefficient_real=_finite(
                    term.get("coeff_re"),
                    label="Qiskit-plateau Pauli coefficient real",
                ),
                coefficient_imaginary=_finite(
                    term.get("coeff_im"),
                    label="Qiskit-plateau Pauli coefficient imaginary",
                ),
                qubit_count=_integer(
                    term.get("nq"),
                    label="Qiskit-plateau Pauli qubit count",
                    minimum=1,
                ),
            )
            for term in (
                _mapping(value, label="Qiskit-plateau runtime term")
                for value in _sequence(
                    block.get("runtime_terms_exyz"),
                    label="Qiskit-plateau runtime terms",
                )
            )
        )
        runtime_start = _integer(
            block.get("runtime_start"),
            label="Qiskit-plateau runtime start",
        )
        runtime_count = _integer(
            block.get("runtime_count"),
            label="Qiskit-plateau runtime count",
            minimum=1,
        )
        if (
            block.get("candidate_label") != labels[logical_index]
            or block.get("logical_index") != logical_index
            or runtime_start != expected_runtime_start
            or runtime_count != len(runtime_terms)
        ):
            raise ReportInputError(
                "Qiskit-plateau parameter block partition drifted"
            )
        operators.append(
            PaperIPrefixOperator(
                candidate_label=str(block["candidate_label"]),
                logical_index=logical_index,
                runtime_start=runtime_start,
                runtime_count=runtime_count,
                execution_mode=str(block.get("execution_mode", "")),
                runtime_terms=runtime_terms,
            )
        )
        expected_runtime_start += runtime_count

    strict_replay = _mapping(
        selected.get("strict_replay"),
        label="Qiskit-plateau selected strict replay",
    )
    ledger = _mapping(
        selected.get("estimator_ledger_receipt"),
        label="Qiskit-plateau selected ledger receipt",
    )
    executed = _mapping(
        ledger.get("cumulative_executed_queries"),
        label="Qiskit-plateau selected executed work",
    )
    component_values = _mapping(
        executed.get("components"),
        label="Qiskit-plateau selected work components",
    )
    components = PaperIWorkComponents(
        n_h_outer=_integer(
            component_values.get("N_H_outer"),
            label="Qiskit-plateau N_H_outer",
        ),
        n_h_refit=_integer(
            component_values.get("N_H_refit"),
            label="Qiskit-plateau N_H_refit",
        ),
        n_grad=_integer(
            component_values.get("N_grad"),
            label="Qiskit-plateau N_grad",
        ),
        n_metric=_integer(
            component_values.get("N_metric"),
            label="Qiskit-plateau N_metric",
        ),
    )
    s_alg = _integer(
        executed.get("S_alg"),
        label="Qiskit-plateau selected S_alg",
    )
    route_contract = _mapping(
        protocol.route_contract,
        label="Qiskit-plateau typed route contract",
    )
    if (
        components.s_alg != s_alg
        or ledger.get("status") != "complete"
        or ledger.get("outer_iteration") != controller_round
        or strict_replay.get("passed") is not True
        or not math.isclose(
            _finite(
                strict_replay.get("fidelity"),
                label="Qiskit-plateau replay fidelity",
            ),
            1.0,
            abs_tol=1.0e-10,
            rel_tol=0.0,
        )
        or selected.get("sr_route_profile")
        != route_contract.get("route_profile")
        or selected.get("sr_route_profile_contract_sha256")
        != route_contract.get("sha256")
    ):
        raise ReportInputError(
            "Qiskit-plateau selected prefix authentication failed"
        )
    logical_parameters = tuple(
        _finite(value, label="Qiskit-plateau logical parameter")
        for value in _sequence(
            selected.get("signed_unwrapped_logical_parameters"),
            label="Qiskit-plateau logical parameters",
        )
    )
    runtime_parameters = tuple(
        _finite(value, label="Qiskit-plateau runtime parameter")
        for value in _sequence(
            selected.get("signed_unwrapped_runtime_parameters"),
            label="Qiskit-plateau runtime parameters",
        )
    )
    if (
        len(logical_parameters) != depth
        or len(runtime_parameters) != expected_runtime_start
    ):
        raise ReportInputError(
            "Qiskit-plateau selected parameter count drifted"
        )
    return PaperIPrefixCompileInput(
        source_method="ra_adapt",
        controller_round=controller_round,
        active_ansatz_depth=depth,
        ordered_operator_labels=labels,
        operators=tuple(operators),
        logical_parameters=logical_parameters,
        runtime_parameters=runtime_parameters,
        reference_state=reference,
        checkpoint_sha256=str(selected.get("checkpoint_sha256", "")),
        projective_state_fingerprint=str(
            selected.get("projective_state_fingerprint", "")
        ),
        problem_request_sha256=str(protocol.problem.problem_request_sha256),
        route_profile=str(route_contract.get("route_profile", "")),
        route_contract_sha256=str(route_contract.get("sha256", "")),
        algorithmic_work=PaperIAlgorithmicWork(
            components=components,
            s_alg=s_alg,
        ),
    )


def _qiskit_plateau_log_projection(log_path: Path) -> list[dict[str, Any]]:
    """Load only the 50 accepted-round progress records from the local run log."""

    rows: list[dict[str, Any]] = []
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReportInputError(
            f"Qiskit-plateau remote-runner log is unreadable: {exc}"
        ) from exc
    for line in lines:
        marker = line.find("AI_LOG ")
        if marker < 0:
            continue
        try:
            payload = json.loads(line[marker + len("AI_LOG ") :])
        except json.JSONDecodeError as exc:
            raise ReportInputError(
                "Qiskit-plateau log contains malformed AI_LOG JSON"
            ) from exc
        if payload.get("event") != "hardcoded_adapt_iter":
            continue
        rows.append(
            {
                "depth": _integer(
                    payload.get("depth"),
                    label="Qiskit-plateau log depth",
                    minimum=1,
                ),
                "energy": _finite(
                    payload.get("energy"),
                    label="Qiskit-plateau log energy",
                ),
                "max_grad": _finite(
                    payload.get("max_grad"),
                    label="Qiskit-plateau log max gradient",
                ),
                "selected_op": str(payload.get("best_op", "")),
                "selected_position": _integer(
                    payload.get("selected_position"),
                    label="Qiskit-plateau log insertion position",
                ),
            }
        )
    if [row["depth"] for row in rows] != list(range(1, 51)):
        raise ReportInputError(
            "Qiskit-plateau log does not contain exactly depths 1 through 50"
        )
    return rows


def _diagnostic_bound_file(
    root: Path,
    binding: Mapping[str, Any],
    *,
    label: str,
    verify_sha256: bool = True,
) -> tuple[Path, dict[str, Any]]:
    relative = PurePosixPath(str(binding.get("path", "")))
    if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
        raise ReportInputError(f"{label} path is unsafe")
    path = root.joinpath(*relative.parts)
    if not path.is_file() or path.is_symlink():
        raise ReportInputError(f"{label} is unavailable")
    expected_size = _integer(
        binding.get("size_bytes"),
        label=f"{label} expected size",
        minimum=1,
    )
    expected_sha256 = str(binding.get("sha256", ""))
    if path.stat().st_size != expected_size:
        raise ReportInputError(f"{label} size drifted")
    observed_sha256 = (
        _sha256_file(path) if verify_sha256 else expected_sha256
    )
    if verify_sha256 and observed_sha256 != expected_sha256:
        raise ReportInputError(f"{label} SHA-256 drifted")
    return path, {
        "path": str(path),
        "sha256": expected_sha256,
        "size_bytes": expected_size,
        "byte_verification": (
            "recomputed_for_report"
            if verify_sha256
            else "bound_by_self_digested_failure_receipt_size_rechecked"
        ),
    }


def _load_qiskit_plateau_macro_diagnostic(
    *,
    run_dir: Path,
    log_path: Path,
    append_cell: Mapping[str, Any],
    compiler: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the summary-failed local run as diagnostic-only evidence."""

    resolved_run_dir = run_dir.resolve()
    resolved_log_path = log_path.resolve()
    if (
        not resolved_run_dir.is_dir()
        or resolved_run_dir.is_symlink()
        or not resolved_log_path.is_file()
        or resolved_log_path.is_symlink()
    ):
        raise ReportInputError(
            "Qiskit-plateau diagnostic run directory or log is unavailable"
        )
    run_manifest_path = resolved_run_dir / "run_manifest.json"
    failure_path = resolved_run_dir / "failure_receipt.json"
    run_manifest, run_manifest_digest = _verified_object(
        run_manifest_path,
        label="Qiskit-plateau local run manifest",
    )
    failure, failure_digest = _verified_object(
        failure_path,
        label="Qiskit-plateau local failure receipt",
    )
    if (
        run_manifest.get("schema") != "paper_i_ra_adapt_local_run_manifest_v1"
        or run_manifest.get("algorithm_id")
        != QISKIT_PLATEAU_MACRO_ALGORITHM_ID
        or run_manifest.get("execution_id")
        != QISKIT_PLATEAU_MACRO_EXECUTION_ID
        or run_manifest.get("candidate_representation")
        != "macro_generator_v1"
        or run_manifest.get("insertion_policy") != "plateau_commutation"
        or run_manifest.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or run_manifest.get("selector_cost_policy")
        != "qiskit_full_trial_ansatz_delta_all_phases_v1"
        or run_manifest.get("selector_cost_phase_reuse")
        != "phase_i_once_then_phase_ii_phase_iii_reuse_v1"
        or run_manifest.get("optimizer") != "powell"
        or run_manifest.get("optimizer_maxiter") != 200
        or run_manifest.get("maximum_controller_rounds") != 50
        or _mapping(
            run_manifest.get("seeds"),
            label="Qiskit-plateau run seeds",
        )
        != {"adapt": 7, "transpiler": 7}
    ):
        raise ReportInputError(
            "Qiskit-plateau local run manifest identity drifted"
        )
    if (
        failure.get("schema")
        != "paper_i_ra_adapt_local_failed_execution_receipt_v1"
        or failure.get("status") != "failed"
        or failure.get("algorithm_id") != QISKIT_PLATEAU_MACRO_ALGORITHM_ID
        or failure.get("maximum_controller_rounds") != 50
        or failure.get("error_type") != "ValueError"
        or failure.get("error_message")
        != (
            "canonical Paper-I summary route identity disagrees with the "
            "typed canonical route authority."
        )
    ):
        raise ReportInputError(
            "Qiskit-plateau post-run failure receipt identity drifted"
        )

    protocol_binding = _mapping(
        run_manifest.get("protocol"),
        label="Qiskit-plateau protocol binding",
    )
    protocol_path = Path(str(protocol_binding.get("path", ""))).resolve()
    try:
        protocol_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ReportInputError(
            "Qiskit-plateau protocol escapes the active repository"
        ) from exc
    if (
        not protocol_path.is_file()
        or protocol_path.is_symlink()
        or protocol_path.stat().st_size
        != _integer(
            protocol_binding.get("size_bytes"),
            label="Qiskit-plateau protocol expected size",
            minimum=1,
        )
        or _sha256_file(protocol_path) != protocol_binding.get("sha256")
    ):
        raise ReportInputError("Qiskit-plateau protocol file binding drifted")
    protocol, protocol_digest = _verified_object(
        protocol_path,
        label="Qiskit-plateau protocol",
    )
    route_contract = _mapping(
        protocol.get("route_contract"),
        label="Qiskit-plateau route contract",
    )
    settings = _mapping(
        route_contract.get("execution_settings"),
        label="Qiskit-plateau execution settings",
    )
    invariants = _mapping(
        route_contract.get("semantic_invariants"),
        label="Qiskit-plateau semantic invariants",
    )
    problem = _mapping(
        protocol.get("problem"),
        label="Qiskit-plateau problem",
    )
    if (
        protocol_digest != protocol_binding.get("canonical_sha256")
        or protocol_digest != failure.get("protocol_sha256")
        or protocol.get("algorithm_id") != QISKIT_PLATEAU_MACRO_ALGORITHM_ID
        or protocol.get("candidate_representation") != "macro_generator_v1"
        or protocol.get("horizon") != 50
        or protocol.get("optimizer") != "powell"
        or protocol.get("optimizer_maxiter") != 200
        or route_contract.get("route_profile")
        != QISKIT_PLATEAU_MACRO_ROUTE_PROFILE
        or settings.get("adapt_insertion_mode")
        != "insertion_commutation_plateau_v1"
        or settings.get("phase3_backend_cost_mode") != "transpile_single_v1"
        or settings.get("phase3_backend_name") != "FakeMarrakesh"
        or settings.get("phase3_backend_optimization_level") != 1
        or settings.get("phase3_backend_transpile_seed") != 7
        or invariants.get("selector_compile_cost_policy")
        != "qiskit_full_trial_ansatz_delta_all_phases_v1"
        or invariants.get("selector_compile_cost_phase_reuse")
        != "phase_i_once_then_phase_ii_phase_iii_reuse_v1"
        or problem.get("problem_request_sha256")
        != "5197b317fe67b5eedabd726e29b897260c18bda9eaf6bc9cc05cf3b0a468b65d"
        or problem.get("n_ph_max") != 3
        or problem.get("total_qubits") != 8
        or not math.isclose(
            _finite(problem.get("u"), label="Qiskit-plateau U"),
            8.0,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        )
    ):
        raise ReportInputError(
            "Qiskit-plateau protocol science identity drifted"
        )

    checkpoint_path, checkpoint_binding = _diagnostic_bound_file(
        resolved_run_dir,
        _mapping(
            failure.get("checkpoint"),
            label="Qiskit-plateau checkpoint binding",
        ),
        label="Qiskit-plateau checkpoint",
    )
    _, ledger_binding = _diagnostic_bound_file(
        resolved_run_dir,
        _mapping(
            failure.get("estimator_ledger"),
            label="Qiskit-plateau estimator ledger binding",
        ),
        label="Qiskit-plateau estimator ledger",
        verify_sha256=False,
    )
    exact_energy = _finite(
        append_cell.get("exact_same_cutoff_energy"),
        label="matched Append same-cutoff ED energy",
    )
    projection = _qiskit_plateau_checkpoint_projection(
        checkpoint_path,
        exact_energy=exact_energy,
    )
    fixed_iteration_qiskit = _fixed_prefix_qiskit_observation(
        _qiskit_plateau_prefix_from_checkpoint(
            checkpoint_path,
            protocol_path=protocol_path,
            controller_round=FIXED_COMPARISON_ROUND,
        ),
        error=float(projection["points"][FIXED_COMPARISON_ROUND]["error"]),
        compiler=compiler,
    )
    log_rows = _qiskit_plateau_log_projection(resolved_log_path)
    for checkpoint_row, log_row in zip(
        projection["history_rows"],
        log_rows,
        strict=True,
    ):
        if (
            checkpoint_row["depth"] != log_row["depth"]
            or checkpoint_row["selected_position"]
            != log_row["selected_position"]
            or checkpoint_row["selected_op"] != log_row["selected_op"]
            or not math.isclose(
                checkpoint_row["energy_before_opt"],
                log_row["energy"],
                abs_tol=1.0e-12,
                rel_tol=1.0e-11,
            )
            or not math.isclose(
                checkpoint_row["max_grad"],
                log_row["max_grad"],
                abs_tol=1.0e-12,
                rel_tol=1.0e-11,
            )
        ):
            raise ReportInputError(
                "Qiskit-plateau checkpoint/log projection drifted"
            )
    projection.pop("history_rows")
    terminal = _mapping(
        projection.get("terminal"),
        label="Qiskit-plateau terminal projection",
    )
    append_terminal = _mapping(
        append_cell.get("terminal"),
        label="matched Append terminal row",
    )
    if (
        append_cell.get("execution_id")
        != "core__strong_weak_u8__nph3__append_macro"
        or append_cell.get("regime") != "strong_weak_u8"
        or append_cell.get("representation") != "macro_generator_v1"
        or append_cell.get("method") != "append"
        or append_terminal.get("status") != "complete"
        or append_terminal.get("k") != 50
    ):
        raise ReportInputError(
            "Qiskit-plateau comparison lacks the matched Append baseline"
        )
    return {
        "schema": "paper_i_ra_qiskit_plateau_vs_append_diagnostic_v1",
        "status": "50_scientific_rounds_post_run_summary_failed",
        "not_paper_evidence": True,
        "execution_id": QISKIT_PLATEAU_MACRO_EXECUTION_ID,
        "algorithm_id": QISKIT_PLATEAU_MACRO_ALGORITHM_ID,
        "regime_id": "strong_weak_u8",
        "candidate_representation": "macro_generator_v1",
        "same_cutoff_exact_energy": exact_energy,
        **projection,
        "fixed_iteration_qiskit": fixed_iteration_qiskit,
        "comparison": {
            "append_execution_id": str(append_cell["execution_id"]),
            "append_terminal": dict(append_terminal),
            "terminal_error_difference_qiskit_ra_minus_append": (
                float(terminal["error"])
                - _finite(
                    append_terminal.get("error"),
                    label="matched Append terminal error",
                )
            ),
            "terminal_error_ratio_qiskit_ra_over_append": (
                float(terminal["error"])
                / max(
                    _finite(
                        append_terminal.get("error"),
                        label="matched Append terminal error",
                    ),
                    1.0e-300,
                )
            ),
            "s_alg_ratio_qiskit_ra_over_append": (
                _integer(
                    terminal.get("S_alg"),
                    label="Qiskit-plateau terminal S_alg",
                )
                / _integer(
                    append_terminal.get("S_alg"),
                    label="matched Append terminal S_alg",
                    minimum=1,
                )
            ),
        },
        "online_qiskit_selector": {
            "backend": "FakeMarrakesh",
            "optimization_level": 1,
            "transpile_seed": 7,
            "cost_mode": "transpile_single_v1",
            "application_scope": "phase1_phase2_phase3_and_fallback_v1",
        },
        "terminal_qiskit_tuple": {
            "status": "unavailable",
            "reason": "canonical_post_run_summary_generation_failed",
            "selector_time_candidate_deltas_substituted": False,
        },
        "source_bindings": {
            "run_manifest": {
                "path": str(run_manifest_path),
                "canonical_sha256": run_manifest_digest,
                "file_sha256": _sha256_file(run_manifest_path),
            },
            "failure_receipt": {
                "path": str(failure_path),
                "canonical_sha256": failure_digest,
                "file_sha256": _sha256_file(failure_path),
            },
            "protocol": {
                "path": str(protocol_path),
                "canonical_sha256": protocol_digest,
                "file_sha256": _sha256_file(protocol_path),
            },
            "checkpoint": checkpoint_binding,
            "estimator_ledger": ledger_binding,
            "remote_runner_log": {
                "path": str(resolved_log_path),
                "sha256": _sha256_file(resolved_log_path),
                "size_bytes": resolved_log_path.stat().st_size,
            },
        },
    }


def _load_qiskit_always_macro_diagnostic(
    *,
    run_dir: Path,
    exact_energy: float,
    compiler: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the completed 13-round Qiskit-ranked always run."""

    resolved_run_dir = run_dir.resolve()
    if not resolved_run_dir.is_dir() or resolved_run_dir.is_symlink():
        raise ReportInputError(
            "Qiskit-always diagnostic run directory is unavailable"
        )
    terminal_path = resolved_run_dir / "terminal_receipt.json"
    terminal, terminal_digest = _verified_object(
        terminal_path,
        label="Qiskit-always terminal receipt",
    )
    if (
        terminal.get("schema")
        != "paper_i_ra_adapt_local_terminal_execution_receipt_v1"
        or terminal.get("status") != "passed"
        or terminal.get("algorithm_id") != QISKIT_ALWAYS_MACRO_ALGORITHM_ID
        or terminal.get("execution_id") != QISKIT_ALWAYS_MACRO_EXECUTION_ID
        or terminal.get("maximum_controller_rounds") != 13
        or terminal.get("accepted_controller_rounds") != 13
        or _mapping(
            terminal.get("stop"),
            label="Qiskit-always terminal stop",
        ).get("primary_reason")
        != "maximum_controller_rounds"
    ):
        raise ReportInputError(
            "Qiskit-always terminal receipt identity drifted"
        )
    artifacts = _mapping(
        terminal.get("artifacts"),
        label="Qiskit-always terminal artifacts",
    )
    loaded: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for role in ("run_manifest", "result", "summary"):
        path, binding = _diagnostic_bound_file(
            resolved_run_dir,
            _mapping(
                artifacts.get(role),
                label=f"Qiskit-always {role} binding",
            ),
            label=f"Qiskit-always {role}",
        )
        loaded[role] = _load_object(path, label=f"Qiskit-always {role}")
        bindings[role] = binding
    run_manifest = loaded["run_manifest"]
    result = loaded["result"]
    summary = loaded["summary"]
    run_manifest_digest = _package_contract().verify_self_digest(
        run_manifest,
        label="Qiskit-always run manifest",
    )
    if (
        run_manifest.get("schema")
        != "paper_i_ra_adapt_local_run_manifest_v1"
        or run_manifest.get("algorithm_id")
        != QISKIT_ALWAYS_MACRO_ALGORITHM_ID
        or run_manifest.get("execution_id")
        != QISKIT_ALWAYS_MACRO_EXECUTION_ID
        or run_manifest.get("candidate_representation")
        != "macro_generator_v1"
        or run_manifest.get("insertion_policy")
        != "always_commutation_reduced"
        or run_manifest.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or run_manifest.get("selector_cost_policy")
        != "qiskit_full_trial_ansatz_delta_all_phases_v1"
        or run_manifest.get("selector_cost_phase_reuse")
        != "phase_i_once_then_phase_ii_phase_iii_reuse_v1"
        or run_manifest.get("maximum_controller_rounds") != 13
        or run_manifest.get("optimizer") != "powell"
        or run_manifest.get("optimizer_maxiter") != 200
        or _mapping(
            run_manifest.get("seeds"),
            label="Qiskit-always run seeds",
        )
        != {"adapt": 7, "transpiler": 7}
    ):
        raise ReportInputError(
            "Qiskit-always run manifest identity drifted"
        )
    bindings["run_manifest"]["canonical_sha256"] = run_manifest_digest

    protocol_binding = _mapping(
        run_manifest.get("protocol"),
        label="Qiskit-always protocol binding",
    )
    protocol_path = Path(str(protocol_binding.get("path", ""))).resolve()
    try:
        protocol_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ReportInputError(
            "Qiskit-always protocol escapes the active repository"
        ) from exc
    if (
        not protocol_path.is_file()
        or protocol_path.is_symlink()
        or protocol_path.stat().st_size
        != _integer(
            protocol_binding.get("size_bytes"),
            label="Qiskit-always protocol size",
            minimum=1,
        )
        or _sha256_file(protocol_path) != protocol_binding.get("sha256")
    ):
        raise ReportInputError("Qiskit-always protocol binding drifted")
    protocol, protocol_digest = _verified_object(
        protocol_path,
        label="Qiskit-always protocol",
    )
    route_contract = _mapping(
        protocol.get("route_contract"),
        label="Qiskit-always route contract",
    )
    settings = _mapping(
        route_contract.get("execution_settings"),
        label="Qiskit-always route settings",
    )
    invariants = _mapping(
        route_contract.get("semantic_invariants"),
        label="Qiskit-always route invariants",
    )
    problem = _mapping(
        protocol.get("problem"),
        label="Qiskit-always problem",
    )
    if (
        protocol_digest != protocol_binding.get("canonical_sha256")
        or protocol_digest != terminal.get("protocol_sha256")
        or protocol.get("algorithm_id") != QISKIT_ALWAYS_MACRO_ALGORITHM_ID
        or protocol.get("horizon") != 13
        or protocol.get("candidate_representation")
        != "macro_generator_v1"
        or route_contract.get("route_profile")
        != QISKIT_ALWAYS_MACRO_ROUTE_PROFILE
        or settings.get("adapt_insertion_mode")
        != "full_commutation_reduced"
        or settings.get("phase3_backend_cost_mode") != "transpile_single_v1"
        or settings.get("phase3_backend_name") != "FakeMarrakesh"
        or settings.get("phase3_backend_optimization_level") != 1
        or settings.get("phase3_backend_transpile_seed") != 7
        or invariants.get("selector_compile_cost_policy")
        != "qiskit_full_trial_ansatz_delta_all_phases_v1"
        or invariants.get("selector_compile_cost_phase_reuse")
        != "phase_i_once_then_phase_ii_phase_iii_reuse_v1"
        or problem.get("problem_request_sha256")
        != "5197b317fe67b5eedabd726e29b897260c18bda9eaf6bc9cc05cf3b0a468b65d"
        or problem.get("n_ph_max") != 3
        or problem.get("total_qubits") != 8
    ):
        raise ReportInputError(
            "Qiskit-always protocol science identity drifted"
        )
    if result.get("schema") != "paper_i_ra_adapt_result_v1":
        raise ReportInputError("Qiskit-always result schema drifted")
    if result.get("protocol") != protocol:
        raise ReportInputError(
            "Qiskit-always result protocol differs from its bound protocol"
        )
    summary_payload = dict(summary)
    if summary_payload.pop("schema", None) != "paper_i_run_summary_v1":
        raise ReportInputError("Qiskit-always summary schema drifted")
    if _without_none(
        _mapping(
            _mapping(
                result.get("run"),
                label="Qiskit-always result run",
            ).get("paper_i_summary"),
            label="Qiskit-always embedded summary",
        )
    ) != _without_none(summary_payload):
        raise ReportInputError("Qiskit-always summary binding drifted")
    trace = _sequence(
        summary.get("accepted_error_trace"),
        label="Qiskit-always accepted error trace",
    )
    transitions = _sequence(
        _mapping(
            result.get("run"),
            label="Qiskit-always result run",
        ).get("accepted_transitions"),
        label="Qiskit-always accepted transitions",
    )
    if len(trace) != 13 or len(transitions) != 13:
        raise ReportInputError(
            "Qiskit-always run does not contain exactly 13 accepted rounds"
        )
    points = [
        {
            "k": 0,
            "energy": _finite(
                _mapping(
                    transitions[0],
                    label="Qiskit-always first transition",
                ).get("energy_before"),
                label="Qiskit-always initial energy",
            ),
        }
    ]
    points[0]["error"] = abs(float(points[0]["energy"]) - exact_energy)
    for expected_round, raw in enumerate(trace, start=1):
        row = _mapping(
            raw,
            label=f"Qiskit-always error trace {expected_round}",
        )
        energy = _finite(
            row.get("accepted_energy"),
            label=f"Qiskit-always round {expected_round} energy",
        )
        error = _finite(
            row.get("absolute_energy_error"),
            label=f"Qiskit-always round {expected_round} error",
        )
        if (
            row.get("controller_round") != expected_round
            or not math.isclose(
                error,
                abs(energy - exact_energy),
                abs_tol=1.0e-11,
                rel_tol=1.0e-10,
            )
        ):
            raise ReportInputError(
                "Qiskit-always accepted error trace drifted"
            )
        points.append(
            {"k": expected_round, "energy": energy, "error": error}
        )
    terminal_energy = _finite(
        terminal.get("final_energy"),
        label="Qiskit-always terminal energy",
    )
    if not math.isclose(
        terminal_energy,
        float(points[-1]["energy"]),
        abs_tol=1.0e-12,
        rel_tol=1.0e-11,
    ):
        raise ReportInputError("Qiskit-always terminal energy drifted")
    terminal_work = _mapping(
        summary.get("canonical_all_work"),
        label="Qiskit-always canonical work",
    )
    fixed_iteration = _fixed_prefix_qiskit_observation(
        _ra_prefix(
            result,
            controller_round=FIXED_COMPARISON_ROUND,
        ),
        error=float(points[FIXED_COMPARISON_ROUND]["error"]),
        compiler=compiler,
    )
    insertion_positions = [
        _integer(
            _mapping(raw, label="Qiskit-always transition").get(
                "insertion_position"
            ),
            label="Qiskit-always insertion position",
        )
        for raw in transitions
    ]
    return {
        "schema": "paper_i_ra_qiskit_always13_diagnostic_v1",
        "status": "13_scientific_rounds_complete",
        "not_paper_evidence": True,
        "execution_id": QISKIT_ALWAYS_MACRO_EXECUTION_ID,
        "algorithm_id": QISKIT_ALWAYS_MACRO_ALGORITHM_ID,
        "regime_id": "strong_weak_u8",
        "candidate_representation": "macro_generator_v1",
        "same_cutoff_exact_energy": exact_energy,
        "points": points,
        "marker": _curve_marker(points, summary=summary),
        "terminal": {
            "k": 13,
            "energy": terminal_energy,
            "error": float(points[-1]["error"]),
            "S_alg": _integer(
                terminal_work.get("s_alg"),
                label="Qiskit-always terminal S_alg",
            ),
        },
        "fixed_iteration_qiskit": fixed_iteration,
        "insertion": {
            "positions": insertion_positions,
            "interior_count": sum(
                position < round_index - 1
                for round_index, position in enumerate(
                    insertion_positions,
                    start=1,
                )
            ),
        },
        "online_qiskit_selector": {
            "backend": "FakeMarrakesh",
            "optimization_level": 1,
            "transpile_seed": 7,
            "cost_mode": "transpile_single_v1",
            "application_scope": "phase1_phase2_phase3_and_fallback_v1",
        },
        "source_bindings": {
            **bindings,
            "terminal_receipt": {
                "path": str(terminal_path),
                "canonical_sha256": terminal_digest,
                "file_sha256": _sha256_file(terminal_path),
            },
            "protocol": {
                "path": str(protocol_path),
                "canonical_sha256": protocol_digest,
                "file_sha256": _sha256_file(protocol_path),
            },
        },
    }


def _stream_verified_singleton_pointer(
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Read the small authenticated resume pointer without loading 2.8 GB."""

    try:
        import ijson
    except ModuleNotFoundError as exc:
        raise ReportInputError(
            "the singleton round-33 diagnostic requires ijson"
        ) from exc
    try:
        with checkpoint_path.open("rb") as stream:
            pointer = next(
                ijson.items(
                    stream,
                    "adapt_vqe.verified_singleton_resume_sidecar",
                ),
                None,
            )
    except (OSError, ValueError) as exc:
        raise ReportInputError(
            f"singleton round-33 resume pointer stream failed: {exc}"
        ) from exc
    return dict(
        _mapping(pointer, label="singleton round-33 resume pointer")
    )


def _load_qiskit_singleton_round33_diagnostic(
    *,
    run_dir: Path,
    ra_cell: Mapping[str, Any],
    append_cell: Mapping[str, Any],
    compiler: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Authenticate and compare the interrupted global-singleton prefix."""

    resolved_run_dir = run_dir.resolve()
    if not resolved_run_dir.is_dir() or resolved_run_dir.is_symlink():
        raise ReportInputError(
            "singleton round-33 diagnostic run directory is unavailable"
        )
    manifest_path = resolved_run_dir / "run_manifest.json"
    manifest, manifest_digest = _verified_object(
        manifest_path,
        label="singleton round-33 run manifest",
    )
    if (
        manifest.get("schema") != "paper_i_ra_adapt_local_run_manifest_v1"
        or manifest.get("algorithm_id")
        != QISKIT_GLOBAL_SINGLETON_ALGORITHM_ID
        or manifest.get("execution_id")
        != QISKIT_GLOBAL_SINGLETON_EXECUTION_ID
        or manifest.get("candidate_representation")
        != "single_pauli_word_v1"
        or manifest.get("insertion_policy") != "plateau_commutation"
        or manifest.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or manifest.get("selector_cost_policy")
        != "qiskit_full_trial_ansatz_delta_all_phases_v1"
        or manifest.get("selector_cost_phase_reuse")
        != "phase_i_once_then_phase_ii_phase_iii_reuse_v1"
        or manifest.get("optimizer") != "powell"
        or manifest.get("optimizer_maxiter") != 200
        or manifest.get("maximum_controller_rounds") != 50
        or _mapping(
            manifest.get("seeds"),
            label="singleton round-33 run seeds",
        )
        != {"adapt": 7, "transpiler": 7}
    ):
        raise ReportInputError(
            "singleton round-33 run manifest identity drifted"
        )

    protocol_binding = _mapping(
        manifest.get("protocol"),
        label="singleton round-33 protocol binding",
    )
    protocol_path = Path(str(protocol_binding.get("path", ""))).resolve()
    try:
        protocol_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ReportInputError(
            "singleton round-33 protocol escapes the active repository"
        ) from exc
    if (
        not protocol_path.is_file()
        or protocol_path.is_symlink()
        or protocol_path.stat().st_size
        != _integer(
            protocol_binding.get("size_bytes"),
            label="singleton round-33 protocol size",
            minimum=1,
        )
        or _sha256_file(protocol_path) != protocol_binding.get("sha256")
    ):
        raise ReportInputError(
            "singleton round-33 protocol file binding drifted"
        )
    protocol, protocol_digest = _verified_object(
        protocol_path,
        label="singleton round-33 protocol",
    )
    route_contract = _mapping(
        protocol.get("route_contract"),
        label="singleton round-33 route contract",
    )
    settings = _mapping(
        route_contract.get("execution_settings"),
        label="singleton round-33 execution settings",
    )
    invariants = _mapping(
        route_contract.get("semantic_invariants"),
        label="singleton round-33 semantic invariants",
    )
    problem = _mapping(
        protocol.get("problem"),
        label="singleton round-33 problem",
    )
    if (
        protocol_digest != protocol_binding.get("canonical_sha256")
        or protocol.get("algorithm_id")
        != QISKIT_GLOBAL_SINGLETON_ALGORITHM_ID
        or protocol.get("candidate_representation")
        != "single_pauli_word_v1"
        or protocol.get("horizon") != 50
        or protocol.get("optimizer") != "powell"
        or protocol.get("optimizer_maxiter") != 200
        or route_contract.get("route_profile")
        != QISKIT_GLOBAL_SINGLETON_ROUTE_PROFILE
        or settings.get("adapt_insertion_mode")
        != "insertion_commutation_plateau_v1"
        or settings.get("phase3_backend_cost_mode")
        != "transpile_single_v1"
        or settings.get("phase3_backend_name") != "FakeMarrakesh"
        or settings.get("phase3_backend_optimization_level") != 1
        or settings.get("phase3_backend_transpile_seed") != 7
        or invariants.get("phase_i_candidate_supply")
        != "global_guarded_singleton_pool_v1"
        or invariants.get("phase_ii_candidate_exposure")
        != "identity_on_retained_singletons_v1"
        or invariants.get("selector_compile_cost_policy")
        != "qiskit_full_trial_ansatz_delta_all_phases_v1"
        or invariants.get("selector_compile_cost_phase_reuse")
        != "phase_i_once_then_phase_ii_phase_iii_reuse_v1"
        or invariants.get("resource_weighting_scope")
        != "all_phase_resource_weighting_v1"
        or problem.get("problem_request_sha256")
        != "e9e9287c677cd2f2af5e9990b2a5742faa225b27fac38f54f7e054ed1fc29a2d"
        or problem.get("n_ph_max") != 7
        or problem.get("total_qubits") != 10
        or not math.isclose(
            _finite(problem.get("u"), label="singleton round-33 U"),
            8.0,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        )
    ):
        raise ReportInputError(
            "singleton round-33 protocol science identity drifted"
        )

    checkpoint_path = resolved_run_dir / "checkpoint.json"
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise ReportInputError(
            "singleton round-33 checkpoint is unavailable or unsafe"
        )
    pointer = _stream_verified_singleton_pointer(checkpoint_path)
    relative_sidecar = PurePosixPath(str(pointer.get("path", "")))
    if (
        relative_sidecar.is_absolute()
        or len(relative_sidecar.parts) != 1
        or "." in relative_sidecar.parts
        or ".." in relative_sidecar.parts
    ):
        raise ReportInputError(
            "singleton round-33 sidecar pointer path is unsafe"
        )
    sidecar_path = resolved_run_dir / relative_sidecar.as_posix()
    if not sidecar_path.is_file() or sidecar_path.is_symlink():
        raise ReportInputError("singleton round-33 sidecar is unavailable")
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar_sha256 = _sha256_bytes(sidecar_bytes)
    try:
        sidecar_raw = json.loads(sidecar_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportInputError(
            "singleton round-33 sidecar is not valid JSON"
        ) from exc
    sidecar = _mapping(sidecar_raw, label="singleton round-33 sidecar")
    controller_state = _mapping(
        sidecar.get("controller_state"),
        label="singleton round-33 controller state",
    )
    selection_state = _mapping(
        sidecar.get("selection_state"),
        label="singleton round-33 selection state",
    )
    source_history = _mapping(
        controller_state.get("source_history_row_evidence"),
        label="singleton round-33 source-history evidence",
    )
    ordered_parent_indices = _sequence(
        selection_state.get("ordered_parent_pool_indices"),
        label="singleton round-33 ordered parent indices",
    )
    feature_counts = _sequence(
        selection_state.get("selected_feature_row_count_per_round"),
        label="singleton round-33 feature-row counts",
    )
    if (
        pointer.get("schema")
        != "static_adapt_verified_singleton_resume_sidecar_pointer_v1"
        or pointer.get("enabled") is not True
        or pointer.get("status") != "complete"
        or pointer.get("sidecar_schema")
        != "static_adapt_signed_active_prefix_resume_sidecar_v2"
        or pointer.get("source_projection_schema")
        != "static_adapt_verified_singleton_resume_source_projection_v1"
        or pointer.get("sha256") != sidecar_sha256
        or sidecar.get("schema") != pointer.get("sidecar_schema")
        or sidecar.get("source_result_digest_scope")
        != pointer.get("source_projection_schema")
        or sidecar.get("source_result_sha256")
        != pointer.get("source_projection_sha256")
        or Path(str(sidecar.get("source_result_json", ""))).resolve()
        != checkpoint_path.resolve()
        or sidecar.get("no_credentials_serialized") is not True
        or controller_state.get("controller_round")
        != MATCHED_SINGLETON_ROUND
        or controller_state.get("source_max_depth")
        != MATCHED_SINGLETON_ROUND
        or source_history.get("depth") != MATCHED_SINGLETON_ROUND
        or selection_state.get("controller_round")
        != MATCHED_SINGLETON_ROUND
        or selection_state.get("pool_size") != 6508
        or len(ordered_parent_indices) != MATCHED_SINGLETON_ROUND
        or feature_counts != [1] * MATCHED_SINGLETON_ROUND
        or sidecar.get("controller_snapshot_sha256")
        != _canonical_sha256(sidecar.get("controller_snapshot"))
        or controller_state.get("source_history_row_evidence_sha256")
        != _canonical_sha256(source_history)
        or selection_state.get("ordered_parent_pool_indices_sha256")
        != _canonical_sha256(ordered_parent_indices)
        or selection_state.get("ordered_logical_candidate_indices_sha256")
        != _canonical_sha256([])
    ):
        raise ReportInputError(
            "singleton round-33 resume-sidecar authentication drifted"
        )

    exact_energy = _finite(
        ra_cell.get("exact_same_cutoff_energy"),
        label="singleton round-33 matched RA exact energy",
    )
    if not math.isclose(
        exact_energy,
        _finite(
            append_cell.get("exact_same_cutoff_energy"),
            label="singleton round-33 matched Append exact energy",
        ),
        abs_tol=1.0e-12,
        rel_tol=0.0,
    ):
        raise ReportInputError(
            "singleton round-33 comparison exact references disagree"
        )
    projection = _qiskit_plateau_checkpoint_projection(
        checkpoint_path,
        exact_energy=exact_energy,
        expected_rounds=MATCHED_SINGLETON_ROUND,
    )
    projection.pop("history_rows")
    current_observation = _fixed_prefix_qiskit_observation(
        _qiskit_plateau_prefix_from_checkpoint(
            checkpoint_path,
            protocol_path=protocol_path,
            controller_round=MATCHED_SINGLETON_ROUND,
        ),
        error=float(projection["points"][MATCHED_SINGLETON_ROUND]["error"]),
        compiler=compiler,
    )
    if (
        ra_cell.get("execution_id")
        != "core__strong_strong_u8__nph7__ra_singleton_plateau"
        or ra_cell.get("regime") != "strong_strong_u8"
        or ra_cell.get("representation") != "single_pauli_word_v1"
        or ra_cell.get("method") != "plateau"
        or append_cell.get("execution_id")
        != "core__strong_strong_u8__nph7__append_singleton"
        or append_cell.get("regime") != "strong_strong_u8"
        or append_cell.get("representation") != "single_pauli_word_v1"
        or append_cell.get("method") != "append"
    ):
        raise ReportInputError(
            "singleton round-33 comparison baselines drifted"
        )
    baseline_ra = dict(
        _mapping(
            ra_cell.get("matched_round_qiskit"),
            label="singleton round-33 staged RA observation",
        )
    )
    baseline_append = dict(
        _mapping(
            append_cell.get("matched_round_qiskit"),
            label="singleton round-33 Append observation",
        )
    )
    if any(
        observation.get("k") != MATCHED_SINGLETON_ROUND
        for observation in (
            current_observation,
            baseline_ra,
            baseline_append,
        )
    ):
        raise ReportInputError(
            "singleton matched-round Qiskit observations drifted"
        )
    return {
        "schema": "paper_i_ra_global_singleton_round33_comparison_v1",
        "status": "33_scientific_rounds_verified_checkpoint",
        "not_paper_evidence": True,
        "execution_id": QISKIT_GLOBAL_SINGLETON_EXECUTION_ID,
        "algorithm_id": QISKIT_GLOBAL_SINGLETON_ALGORITHM_ID,
        "regime_id": "strong_strong_u8",
        "candidate_representation": "single_pauli_word_v1",
        "same_cutoff_exact_energy": exact_energy,
        "matched_round": MATCHED_SINGLETON_ROUND,
        "baseline_execution_ids": {
            "staged_proxy_ra_plateau": str(ra_cell["execution_id"]),
            "conventional_append_adapt": str(append_cell["execution_id"]),
        },
        "points": list(projection["points"]),
        "matched_observations": {
            "global_singleton_qiskit_ra": current_observation,
            "staged_proxy_ra_plateau": baseline_ra,
            "conventional_append_adapt": baseline_append,
        },
        "baseline_curves": {
            "staged_proxy_ra_plateau": list(ra_cell["points"])[
                : MATCHED_SINGLETON_ROUND + 1
            ],
            "conventional_append_adapt": list(append_cell["points"])[
                : MATCHED_SINGLETON_ROUND + 1
            ],
        },
        "accounting": dict(
            _mapping(
                projection.get("accounting"),
                label="singleton round-33 accounting",
            )
        ),
        "insertion": dict(
            _mapping(
                projection.get("insertion"),
                label="singleton round-33 insertion",
            )
        ),
        "checkpoint_state": dict(
            _mapping(
                projection.get("checkpoint_state"),
                label="singleton round-33 checkpoint state",
            )
        ),
        "online_qiskit_selector": {
            "backend": "FakeMarrakesh",
            "optimization_level": 1,
            "transpile_seed": 7,
            "cost_mode": "transpile_single_v1",
            "application_scope": "phase1_phase2_phase3_and_fallback_v1",
            "phase_i_candidate_supply": "global_guarded_singleton_pool_v1",
        },
        "source_bindings": {
            "run_manifest": {
                "path": str(manifest_path),
                "canonical_sha256": manifest_digest,
                "file_sha256": _sha256_file(manifest_path),
            },
            "protocol": {
                "path": str(protocol_path),
                "canonical_sha256": protocol_digest,
                "file_sha256": _sha256_file(protocol_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "file_sha256": _sha256_file(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "source_projection_sha256": pointer[
                    "source_projection_sha256"
                ],
            },
            "resume_sidecar": {
                "path": str(sidecar_path),
                "file_sha256": sidecar_sha256,
                "size_bytes": sidecar_path.stat().st_size,
            },
        },
    }


def _extract_ra_cell(
    *,
    execution_id: str,
    job: Mapping[str, Any],
    result: Mapping[str, Any],
    summary: Mapping[str, Any],
    closure: Mapping[str, Any],
    exact_energy: float,
    compiler: Callable[[Any], Any] | None,
) -> dict[str, Any]:
    if result.get("schema") != "paper_i_ra_adapt_result_v1":
        raise ReportInputError(f"{execution_id}: RA result schema drifted")
    if summary.get("schema") != "paper_i_run_summary_v1":
        raise ReportInputError(f"{execution_id}: RA summary schema drifted")
    trace = _sequence(
        summary.get("accepted_error_trace"), label=f"{execution_id} RA trace"
    )
    if len(trace) != 50:
        raise ReportInputError(f"{execution_id}: RA trace is not 50 rounds")
    provenance = _mapping(
        summary.get("provenance"), label=f"{execution_id} RA provenance"
    )
    if not math.isclose(
        _finite(
            provenance.get("exact_same_cutoff_energy"),
            label=f"{execution_id} summary E_ED",
        ),
        exact_energy,
        abs_tol=1.0e-10,
        rel_tol=0.0,
    ):
        raise ReportInputError(f"{execution_id}: RA same-cutoff reference drifted")
    points: list[dict[str, Any]] = []
    for expected_round, raw in enumerate(trace, start=1):
        row = _mapping(raw, label=f"{execution_id} RA trace row")
        round_index = _integer(
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
        )
        if round_index != expected_round or not math.isclose(
            error,
            abs(energy - exact_energy),
            abs_tol=1.0e-12,
            rel_tol=1.0e-11,
        ):
            raise ReportInputError(f"{execution_id}: RA trace math drifted")
        points.append({"k": round_index, "error": error})
    run = _mapping(result.get("run"), label=f"{execution_id} RA run")
    transitions = _sequence(
        run.get("accepted_transitions"),
        label=f"{execution_id} RA transitions",
    )
    initial = _finite(
        _mapping(transitions[0], label=f"{execution_id} first transition").get(
            "energy_before"
        ),
        label=f"{execution_id} initial energy",
    )
    points.insert(0, {"k": 0, "error": abs(initial - exact_energy)})
    all_work = _mapping(
        summary.get("canonical_all_work"),
        label=f"{execution_id} RA all-work",
    )
    s_alg = _integer(
        all_work.get("s_alg"), label=f"{execution_id} RA S_alg"
    )
    prefix_work = _sequence(
        _mapping(
            run.get("canonical_reporting"),
            label=f"{execution_id} canonical reporting",
        ).get("accepted_prefix_work"),
        label=f"{execution_id} accepted-prefix work",
    )
    g10 = _mapping(
        _mapping(
            _mapping(closure.get("gates"), label=f"{execution_id} gates").get(
                "G10"
            ),
            label=f"{execution_id} G10",
        ).get("evidence"),
        label=f"{execution_id} G10 evidence",
    )
    if (
        len(transitions) != 50
        or len(prefix_work) != 50
        or _mapping(prefix_work[-1], label="RA terminal work").get("s_alg")
        != s_alg
        or _mapping(transitions[-1], label="RA terminal transition").get(
            "cumulative_s_alg"
        )
        != s_alg
        or g10.get("S_alg") != s_alg
    ):
        raise ReportInputError(f"{execution_id}: RA terminal S_alg drifted")
    prefix = _ra_terminal_prefix(result, s_alg=s_alg)
    resources, checkpoint_sha, compile_payload = _compile_terminal_qiskit(
        prefix, compiler=compiler
    )
    fixed_prefix = None
    if execution_id in FIXED_COMPARISON_EXECUTION_IDS:
        fixed_prefix = _fixed_prefix_qiskit_observation(
            _ra_prefix(
                result,
                controller_round=FIXED_COMPARISON_ROUND,
            ),
            error=float(points[FIXED_COMPARISON_ROUND]["error"]),
            compiler=compiler,
        )
    matched_round_prefix = None
    if execution_id in MATCHED_SINGLETON_EXECUTION_IDS:
        matched_round_prefix = _fixed_prefix_qiskit_observation(
            _ra_prefix(
                result,
                controller_round=MATCHED_SINGLETON_ROUND,
            ),
            error=float(points[MATCHED_SINGLETON_ROUND]["error"]),
            compiler=compiler,
        )
    terminal = {
        "k": 50,
        "error": points[-1]["error"],
        **resources,
        "S_alg": s_alg,
        "status": "complete",
    }
    return {
        "execution_id": execution_id,
        "regime": str(job["regime_id"]),
        "representation": str(job["candidate_representation"]),
        "method": _method_key(str(job["route_id"])),
        "points": points,
        "marker": _curve_marker(points, summary=summary),
        "terminal": terminal,
        "exact_same_cutoff_energy": exact_energy,
        "terminal_checkpoint_sha256": checkpoint_sha,
        "terminal_compile_convention": "table_i_basis_gate_transpile_v1",
        "terminal_compile_source": (
            "common_typed_terminal_prefix_recompile_v1"
        ),
        "serialized_terminal_cross_check": "not_applicable",
        "terminal_qiskit_version": compile_payload.get("qiskit_version"),
        "terminal_generator_coefficients_sha256": compile_payload.get(
            "generator_coefficients_sha256"
        ),
        "fixed_iteration_qiskit": fixed_prefix,
        "matched_round_qiskit": matched_round_prefix,
    }


def _append_problem_from_protocol(protocol: Any) -> Any:
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
    if ResolvedProblemReceipt.from_problem(problem) != receipt:
        raise ReportInputError(
            "Append reconstructed problem drifted from its protocol receipt"
        )
    return problem


def _append_protocol_for_reporting(
    *,
    job: Mapping[str, Any],
    expected_protocol: Mapping[str, Any],
) -> Any:
    """Rehydrate the job-bound Append protocol for read-only reporting."""

    from pipelines.static_adapt.ra_adapt.contracts import (
        load_resolved_ra_adapt_protocol,
    )

    protocol_binding = _mapping(
        job.get("protocol"), label="Append job protocol binding"
    )
    protocol_relative = PurePosixPath(str(protocol_binding.get("path", "")))
    if (
        protocol_relative.is_absolute()
        or "." in protocol_relative.parts
        or ".." in protocol_relative.parts
    ):
        raise ReportInputError("Append job protocol path is unsafe")
    # The direct typed loader is the contracts module's inspection surface.
    # It digest-checks this one protocol without granting execution authority
    # or reinterpreting unrelated cells through the ambient route registry.
    protocol = load_resolved_ra_adapt_protocol(
        REPO_ROOT.joinpath(*protocol_relative.parts)
    )
    if protocol.to_dict() != expected_protocol:
        raise ReportInputError("Append typed protocol reconstruction drifted")
    return protocol


def _append_prefix(
    result: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    controller_round: int,
    expected_s_alg: int | None = None,
) -> Any:
    """Reconstruct one authenticated Append accepted-prefix compiler input."""

    if result.get("schema") != "paper_i_append_adapt_result_v1":
        raise ReportInputError("Append result schema drifted")
    expected_protocol = _protocol_for_job(job)
    embedded_protocol = _mapping(
        result.get("protocol"), label="Append embedded protocol"
    )
    if embedded_protocol != expected_protocol:
        raise ReportInputError(
            "Append result protocol differs from the job-bound protocol"
        )

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
        validate_controller_replay_evidence,
    )

    protocol = _append_protocol_for_reporting(
        job=job,
        expected_protocol=expected_protocol,
    )
    problem = _append_problem_from_protocol(protocol)
    (
        _request,
        _parent_inventory,
        executable_inventory,
        _lineage,
    ) = _validate_resolved_append_protocol(problem, protocol)

    payload = _mapping(
        result.get("result_payload"), label="Append result payload"
    )
    if (
        payload.get("schema") != "paper_i_append_adapt_execution_v1"
        or payload.get("protocol_sha256") != protocol.sha256
        or payload.get("candidate_representation")
        != protocol.candidate_representation
        or payload.get("executable_pool") != protocol.executable_pool.to_dict()
        or result.get("executable_pool")
        != protocol.executable_pool.to_dict()
    ):
        raise ReportInputError("Append result/pool protocol binding drifted")
    replay = validate_controller_replay_evidence(
        payload.get("controller_replay_evidence")
    )
    signed_prefixes = _sequence(
        replay.get("signed_controller_round_prefixes"),
        label="Append signed controller prefixes",
    )
    history = _sequence(payload.get("history"), label="Append result history")
    if (
        replay.get("method_family") != "append_adapt"
        or replay.get("protocol_sha256") != protocol.sha256
        or replay.get("problem_request_sha256")
        != protocol.problem.problem_request_sha256
        or len(signed_prefixes) != 50
        or len(history) != 50
    ):
        raise ReportInputError("Append replay horizon/binding drifted")
    round_index = _integer(
        controller_round,
        label="Append prefix controller round",
        minimum=1,
    )
    if round_index > 50:
        raise ReportInputError(
            "Append prefix controller round exceeds the horizon"
        )
    zero_index = round_index - 1
    selected_signed = _mapping(
        signed_prefixes[zero_index], label="Append selected signed prefix"
    )
    checkpoint = _mapping(
        selected_signed.get("active_prefix_checkpoint"),
        label="Append selected active-prefix checkpoint",
    )
    history_checkpoint = _mapping(
        _mapping(history[zero_index], label="Append selected history row").get(
            "active_prefix_checkpoint"
        ),
        label="Append selected history checkpoint",
    )
    if checkpoint != history_checkpoint:
        raise ReportInputError(
            "Append selected replay/history checkpoint drifted"
        )
    labels = tuple(
        str(value)
        for value in _sequence(
            checkpoint.get("accepted_operator_labels"),
            label="Append selected accepted labels",
        )
    )
    identities = tuple(
        str(value)
        for value in _sequence(
            checkpoint.get("accepted_generator_identities"),
            label="Append selected accepted generator identities",
        )
    )
    logical_parameters = tuple(
        _finite(value, label="Append selected logical parameter")
        for value in _sequence(
            checkpoint.get("logical_parameters"),
            label="Append selected logical parameters",
        )
    )
    runtime_parameters = tuple(
        _finite(value, label="Append selected runtime parameter")
        for value in _sequence(
            checkpoint.get("runtime_parameters"),
            label="Append selected runtime parameters",
        )
    )
    if (
        selected_signed.get("controller_round") != round_index
        or checkpoint.get("controller_round") != round_index
        or checkpoint.get("protocol_sha256") != protocol.sha256
        or checkpoint.get("problem_request_sha256")
        != protocol.problem.problem_request_sha256
        or len(labels) != round_index
        or len(identities) != round_index
        or len(logical_parameters) != round_index
    ):
        raise ReportInputError("Append selected checkpoint lineage drifted")
    if round_index == 50 and (
        tuple(payload.get("accepted_operator_labels", ())) != labels
        or tuple(payload.get("accepted_generator_identities", ()))
        != identities
        or tuple(payload.get("logical_theta", ())) != logical_parameters
        or tuple(payload.get("runtime_theta", ())) != runtime_parameters
    ):
        raise ReportInputError("Append terminal payload lineage drifted")

    candidates: dict[str, Any] = {}
    for candidate in executable_inventory.candidates:
        label = str(candidate.label)
        if label in candidates:
            raise ReportInputError(
                f"Append executable pool duplicates label {label!r}"
            )
        candidates[label] = candidate
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
            raise ReportInputError(
                "Append terminal accepted generator is absent from the "
                "protocol-locked executable pool"
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
                    label="Append Pauli qubit count",
                    minimum=1,
                ),
            )
            for term in (
                _mapping(value, label="Append serialized candidate term")
                for value in candidate.serialized_terms_exyz
            )
        )
        if not terms:
            raise ReportInputError("Append candidate has no runtime terms")
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
        raise ReportInputError(
            "Append selected operators do not partition runtime parameters"
        )

    reference_array = np.asarray(
        problem.reference_state.build_state(), dtype=complex
    ).reshape(-1)
    reference_norm = float(np.linalg.norm(reference_array))
    if not math.isclose(
        reference_norm, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise ReportInputError("Append reference state is not normalized")
    reference_array = reference_array / reference_norm
    reference = PaperIReferenceState(
        amplitudes_real=tuple(float(value.real) for value in reference_array),
        amplitudes_imaginary=tuple(
            float(value.imag) for value in reference_array
        ),
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
        label="Append selected executed-query components",
    )
    components = PaperIWorkComponents(
        n_h_outer=_integer(
            components_raw.get("N_H_outer"), label="Append N_H_outer"
        ),
        n_h_refit=_integer(
            components_raw.get("N_H_refit"), label="Append N_H_refit"
        ),
        n_grad=_integer(
            components_raw.get("N_grad"), label="Append N_grad"
        ),
        n_metric=_integer(
            components_raw.get("N_metric"), label="Append N_metric"
        ),
    )
    checkpoint_s_alg = _integer(
        executed.get("S_alg"), label="Append checkpoint S_alg"
    )
    if (
        components.s_alg != checkpoint_s_alg
        or (
            expected_s_alg is not None
            and checkpoint_s_alg != expected_s_alg
        )
    ):
        raise ReportInputError("Append selected checkpoint S_alg drifted")
    route_contract = _mapping(
        protocol.route_contract, label="Append route contract"
    )
    return PaperIPrefixCompileInput(
        source_method="append_adapt",
        controller_round=round_index,
        active_ansatz_depth=len(labels),
        ordered_operator_labels=labels,
        operators=tuple(operators),
        logical_parameters=logical_parameters,
        runtime_parameters=runtime_parameters,
        reference_state=reference,
        checkpoint_sha256=str(checkpoint.get("checkpoint_sha256", "")),
        projective_state_fingerprint=str(
            checkpoint.get("projective_state_fingerprint", "")
        ),
        problem_request_sha256=str(
            protocol.problem.problem_request_sha256
        ),
        route_profile=str(route_contract.get("route_profile", "")),
        route_contract_sha256=str(route_contract.get("sha256", "")),
        algorithmic_work=PaperIAlgorithmicWork(
            components=components,
            s_alg=checkpoint_s_alg,
        ),
    )


def _append_terminal_prefix(
    result: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    s_alg: int,
) -> Any:
    """Backward-compatible round-50 Append prefix reconstruction."""

    return _append_prefix(
        result,
        job=job,
        controller_round=50,
        expected_s_alg=s_alg,
    )


def _crosscheck_append_terminal_qiskit(
    *,
    fresh: Mapping[str, Any],
    fresh_payload: Mapping[str, Any],
    serialized_payload: Mapping[str, Any],
) -> None:
    try:
        serialized = qiskit_cost_fields(serialized_payload)
    except (TypeError, ValueError) as exc:
        raise ReportInputError(
            f"Append serialized terminal Qiskit cost is unavailable: {exc}"
        ) from exc
    for field in (
        "N2q",
        "D2q",
        "Dc",
        "W1q",
        "B1q",
        "qiskit_basis_work_status",
        "qiskit_basis_work_schema",
    ):
        if fresh.get(field) != serialized.get(field):
            raise ReportInputError(
                f"serialized terminal Qiskit {field} mismatch"
            )
    for field in (
        "compile_convention",
        "compiled_basis_gates",
        "compiled_circuit_scope",
        "generator_coefficients_sha256",
        "logical_operator_count",
        "qiskit_transpile_optimization_level",
        "qiskit_transpile_seed",
        "qiskit_version",
        "runtime_rotation_count",
    ):
        if (
            field in serialized_payload
            and fresh_payload.get(field) != serialized_payload.get(field)
        ):
            raise ReportInputError(
                f"serialized terminal Qiskit {field} mismatch"
            )


def _extract_append_cell(
    *,
    execution_id: str,
    job: Mapping[str, Any],
    result: Mapping[str, Any],
    summary: Mapping[str, Any],
    closure: Mapping[str, Any],
    exact_energy: float,
    compiler: Callable[[Any], Any] | None,
) -> dict[str, Any]:
    if summary.get("schema") != "paper_i_append_run_summary_v1":
        raise ReportInputError(f"{execution_id}: Append summary schema drifted")
    history = _sequence(
        summary.get("accepted_history"),
        label=f"{execution_id} Append history",
    )
    if (
        len(history) != 50
        or summary.get("controller_rounds_completed") != 50
        or summary.get("protocol_horizon") != 50
        or summary.get("stop_reason") != "maximum_controller_rounds"
    ):
        raise ReportInputError(f"{execution_id}: Append horizon/stop drifted")
    points: list[dict[str, Any]] = []
    initial: float | None = None
    for expected_round, raw in enumerate(history, start=1):
        row = _mapping(raw, label=f"{execution_id} Append history row")
        round_index = _integer(
            row.get("controller_round"),
            label=f"{execution_id} Append round",
            minimum=1,
        )
        if round_index != expected_round:
            raise ReportInputError(f"{execution_id}: Append rounds drifted")
        before = _finite(
            row.get("energy_before"),
            label=f"{execution_id} Append energy before",
        )
        after = _finite(
            row.get("energy_after"),
            label=f"{execution_id} Append energy after",
        )
        if initial is None:
            initial = before
        points.append({"k": round_index, "error": abs(after - exact_energy)})
    if initial is None:
        raise ReportInputError(f"{execution_id}: Append history is empty")
    points.insert(0, {"k": 0, "error": abs(initial - exact_energy)})
    final_energy = _finite(
        summary.get("final_energy"), label=f"{execution_id} Append final energy"
    )
    if not math.isclose(
        points[-1]["error"],
        abs(final_energy - exact_energy),
        abs_tol=1.0e-12,
        rel_tol=1.0e-11,
    ):
        raise ReportInputError(f"{execution_id}: Append terminal energy drifted")
    accounting = _mapping(
        summary.get("estimator_accounting"),
        label=f"{execution_id} Append accounting",
    )
    s_alg = _integer(
        accounting.get("S_alg"), label=f"{execution_id} Append S_alg"
    )
    g10 = _mapping(
        _mapping(
            _mapping(closure.get("gates"), label=f"{execution_id} gates").get(
                "G10"
            ),
            label=f"{execution_id} G10",
        ).get("evidence"),
        label=f"{execution_id} G10 evidence",
    )
    if g10.get("S_alg") != s_alg:
        raise ReportInputError(f"{execution_id}: Append S_alg drifted")
    if result.get("paper_i_summary") != summary:
        raise ReportInputError(
            f"{execution_id}: Append embedded summary drifted"
        )
    resources = _mapping(
        summary.get("resources"), label=f"{execution_id} Append resources"
    )
    serialized_compiled = _mapping(
        resources.get("terminal_compiled_resources"),
        label=f"{execution_id} Append terminal resources",
    )
    if (
        resources.get("terminal_observation_status") != "ok"
        or serialized_compiled.get("compiled_circuit_stats_status") != "ok"
    ):
        raise ReportInputError(
            f"{execution_id}: Append terminal compilation unavailable"
        )
    try:
        prefix = _append_terminal_prefix(result, job=job, s_alg=s_alg)
        qiskit, checkpoint_sha, compile_payload = (
            _compile_terminal_qiskit(prefix, compiler=compiler)
        )
        _crosscheck_append_terminal_qiskit(
            fresh=qiskit,
            fresh_payload=compile_payload,
            serialized_payload=serialized_compiled,
        )
    except ReportInputError:
        raise
    except Exception as exc:
        raise ReportInputError(
            f"{execution_id}: Append terminal prefix reconstruction or "
            f"compilation failed: {exc}"
        ) from exc
    terminal = {
        "k": 50,
        "energy": final_energy,
        "error": points[-1]["error"],
        **qiskit,
        "S_alg": s_alg,
        "status": "complete",
    }
    fixed_prefix = None
    if execution_id in FIXED_COMPARISON_EXECUTION_IDS:
        fixed_prefix = _fixed_prefix_qiskit_observation(
            _append_prefix(
                result,
                job=job,
                controller_round=FIXED_COMPARISON_ROUND,
            ),
            error=float(points[FIXED_COMPARISON_ROUND]["error"]),
            compiler=compiler,
        )
    matched_round_prefix = None
    if execution_id in MATCHED_SINGLETON_EXECUTION_IDS:
        matched_round_prefix = _fixed_prefix_qiskit_observation(
            _append_prefix(
                result,
                job=job,
                controller_round=MATCHED_SINGLETON_ROUND,
            ),
            error=float(points[MATCHED_SINGLETON_ROUND]["error"]),
            compiler=compiler,
        )
    return {
        "execution_id": execution_id,
        "regime": str(job["regime_id"]),
        "representation": str(job["candidate_representation"]),
        "method": "append",
        "points": points,
        "marker": _curve_marker(points, summary=summary),
        "terminal": terminal,
        "exact_same_cutoff_energy": exact_energy,
        "terminal_checkpoint_sha256": checkpoint_sha,
        "terminal_compile_convention": "table_i_basis_gate_transpile_v1",
        "terminal_compile_source": (
            "common_typed_terminal_prefix_recompile_v1"
        ),
        "serialized_terminal_cross_check": "passed",
        "terminal_qiskit_version": compile_payload.get("qiskit_version"),
        "terminal_generator_coefficients_sha256": compile_payload.get(
            "generator_coefficients_sha256"
        ),
        "fixed_iteration_qiskit": fixed_prefix,
        "matched_round_qiskit": matched_round_prefix,
    }


def _load_validated_attempt_cell(
    *,
    execution_id: str,
    expected_job: Mapping[str, Any],
    attempt_relative_path: str,
    expected_attempt_sha256: str,
    expected_worker_receipt_sha256: str,
    fetched_dir: Path,
    terminal_qiskit_compiler: Callable[[Any], Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one already-validated attempt and recheck report-facing closure."""

    attempt = _safe_attempt_path(fetched_dir, attempt_relative_path)
    attempt_sha = _sha256_file(attempt)
    if attempt_sha != expected_attempt_sha256:
        raise ReportInputError(f"{execution_id}: selected archive hash drifted")
    loaded = _load_attempt_payloads(
        attempt,
        execution_id=execution_id,
    )
    job, job_file_sha, job_size = loaded["job"]
    worker, worker_file_sha, worker_size = loaded["worker"]
    manifest, manifest_file_sha, manifest_size = loaded["manifest"]
    result, result_file_sha, result_size = loaded["result"]
    summary, summary_file_sha, summary_size = loaded["summary"]
    if job != expected_job:
        raise ReportInputError(f"{execution_id}: archived job spec drifted")
    try:
        _package_contract().verify_self_digest(
            job, label=f"{execution_id} archived job spec"
        )
        worker_sha = _package_contract().verify_self_digest(
            worker, label=f"{execution_id} worker receipt"
        )
    except Exception as exc:
        raise ReportInputError(
            f"{execution_id}: archived authority self-digest failed: {exc}"
        ) from exc
    if (
        worker_sha != expected_worker_receipt_sha256
        or worker.get("sha256") != expected_worker_receipt_sha256
    ):
        raise ReportInputError(f"{execution_id}: worker receipt digest drifted")
    bindings = _artifact_binding_map(worker)
    for role, digest, size in (
        ("execution_manifest", manifest_file_sha, manifest_size),
        ("result", result_file_sha, result_size),
        ("summary", summary_file_sha, summary_size),
    ):
        _verify_loaded_artifact(bindings, role, digest, size)
    closure, exact_energy = _closure_context(
        execution_id=execution_id,
        job=job,
        worker=worker,
        manifest=manifest,
    )
    if job.get("execution_entrypoint") == "run_append_adapt":
        cell = _extract_append_cell(
            execution_id=execution_id,
            job=job,
            result=result,
            summary=summary,
            closure=closure,
            exact_energy=exact_energy,
            compiler=terminal_qiskit_compiler,
        )
    elif job.get("execution_entrypoint") == "run_ra_adapt":
        cell = _extract_ra_cell(
            execution_id=execution_id,
            job=job,
            result=result,
            summary=summary,
            closure=closure,
            exact_energy=exact_energy,
            compiler=terminal_qiskit_compiler,
        )
    else:
        raise ReportInputError(f"{execution_id}: unknown execution entrypoint")
    if len(cell["points"]) != 51:
        raise ReportInputError(f"{execution_id}: report curve is not 0..50")
    return cell, {
        "execution_id": execution_id,
        "attempt_path": attempt_relative_path,
        "attempt_sha256": attempt_sha,
        "job_file_sha256": job_file_sha,
        "job_size_bytes": job_size,
        "worker_receipt_file_sha256": worker_file_sha,
        "worker_receipt_size_bytes": worker_size,
        "worker_receipt_sha256": worker["sha256"],
        "execution_manifest_file_sha256": manifest_file_sha,
        "result_file_sha256": result_file_sha,
        "summary_file_sha256": summary_file_sha,
        "regime_id": job["regime_id"],
        "route_id": job["route_id"],
        "candidate_representation": job["candidate_representation"],
        "nph": job["nph"],
        "exact_same_cutoff_energy": exact_energy,
        "plotted_point_count": 51,
        "marker": dict(cell["marker"]),
        "terminal": cell["terminal"],
        "terminal_checkpoint_sha256": cell[
            "terminal_checkpoint_sha256"
        ],
        "terminal_compile_source": cell["terminal_compile_source"],
        "serialized_terminal_cross_check": cell[
            "serialized_terminal_cross_check"
        ],
        "terminal_qiskit_version": cell.get("terminal_qiskit_version"),
        "terminal_generator_coefficients_sha256": cell.get(
            "terminal_generator_coefficients_sha256"
        ),
    }


def load_selected_cells(
    *,
    selection_path: Path,
    validation_path: Path,
    fetched_dir: Path,
    terminal_qiskit_compiler: Callable[[Any], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_jobs = _expected_jobs()
    selection, selection_sha = _verified_object(
        selection_path, label="explicit attempt selection"
    )
    validation, validation_sha = _verified_object(
        validation_path, label="fetched validation"
    )
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("package_id") != PACKAGE_ID
        or selection.get("selected_count") != 48
        or selection.get("automatic_attempt_selection_performed") is not False
        or selection.get("paper_evidence_adopted") is not False
        or selection.get("fetched_validation_sha256") != validation_sha
        or validation.get("schema") != VALIDATION_SCHEMA
        or validation.get("package_id") != PACKAGE_ID
        or validation.get("automatic_attempt_selection_performed") is not False
        or validation.get("paper_evidence_adopted") is not False
    ):
        raise ReportInputError("selection/validation authority drifted")
    attempts = {
        (str(row.get("execution_id")), str(row.get("sha256"))): row
        for row in _sequence(
            validation.get("attempts"), label="validated attempts"
        )
        if isinstance(row, Mapping) and row.get("status") == "passed"
    }
    selected_rows = _sequence(
        selection.get("selected_attempts"), label="selected attempts"
    )
    selected_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in selected_rows:
        row = _mapping(raw, label="selected attempt")
        execution_id = str(row.get("execution_id", ""))
        if execution_id in selected_by_id:
            raise ReportInputError(f"duplicate selected cell: {execution_id}")
        attempt = attempts.get(
            (execution_id, str(row.get("attempt_sha256", "")))
        )
        if (
            execution_id not in expected_jobs
            or attempt is None
            or attempt.get("path") != row.get("attempt_path")
            or attempt.get("worker_receipt_sha256")
            != row.get("worker_receipt_sha256")
        ):
            raise ReportInputError(
                f"{execution_id}: selected attempt is not validated/passed"
            )
        selected_by_id[execution_id] = row
    if set(selected_by_id) != set(expected_jobs):
        missing = sorted(set(expected_jobs).difference(selected_by_id))
        raise ReportInputError(
            f"explicit selection does not cover all 48 cells: {missing}"
        )

    cells: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for execution_id in sorted(expected_jobs):
        selected = selected_by_id[execution_id]
        relative = str(selected.get("attempt_path", ""))
        cell, source_row = _load_validated_attempt_cell(
            execution_id=execution_id,
            expected_job=expected_jobs[execution_id],
            attempt_relative_path=relative,
            expected_attempt_sha256=str(selected.get("attempt_sha256", "")),
            expected_worker_receipt_sha256=str(
                selected.get("worker_receipt_sha256", "")
            ),
            fetched_dir=fetched_dir,
            terminal_qiskit_compiler=terminal_qiskit_compiler,
        )
        cells.append(cell)
        source_rows.append(source_row)
    return cells, {
        "selection": {
            "path": str(selection_path),
            "sha256": selection_sha,
            "file_sha256": _sha256_file(selection_path),
        },
        "validation": {
            "path": str(validation_path),
            "sha256": validation_sha,
            "file_sha256": _sha256_file(validation_path),
        },
        "selected_sources": source_rows,
    }


def load_partial_cells(
    *,
    validation_path: Path,
    fetched_dir: Path,
    terminal_qiskit_compiler: Callable[[Any], Any] | None = None,
    method_family: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a strict, unambiguous subset of passed validated attempts.

    This diagnostic path performs no choice between successful retries. A cell
    is included only when the validation receipt contains exactly one passed
    attempt for that execution id. A complete 48-cell validation remains the
    domain of explicit-selection final mode.
    """

    expected_jobs = _expected_jobs()
    validation, validation_sha = _verified_object(
        validation_path, label="fetched validation"
    )
    raw_attempts = _sequence(
        validation.get("attempts"), label="validated attempts"
    )
    if (
        validation.get("schema") != VALIDATION_SCHEMA
        or validation.get("package_id") != PACKAGE_ID
        or validation.get("automatic_attempt_selection_performed") is not False
        or validation.get("paper_evidence_adopted") is not False
        or validation.get("status") != "validated_no_selection"
        or validation.get("attempt_count") != len(raw_attempts)
    ):
        raise ReportInputError("partial validation authority drifted")

    if method_family not in {None, "append", "ra"}:
        raise ReportInputError(
            f"unknown partial source method family: {method_family}"
        )
    all_passed_by_id: dict[str, list[Mapping[str, Any]]] = {}
    excluded_nonpassed_execution_ids: set[str] = set()
    for raw in raw_attempts:
        row = _mapping(raw, label="validated attempt")
        execution_id = str(row.get("execution_id", ""))
        if execution_id not in expected_jobs:
            raise ReportInputError(
                f"validated attempt is outside the package matrix: "
                f"{execution_id}"
            )
        if row.get("status") == "passed":
            all_passed_by_id.setdefault(execution_id, []).append(row)
        elif (
            method_family is None
            or (
                "append"
                if _method_key(
                    str(expected_jobs[execution_id]["route_id"])
                )
                == "append"
                else "ra"
            )
            == method_family
        ):
            excluded_nonpassed_execution_ids.add(execution_id)
    observed_passed_ids = sorted(all_passed_by_id)
    if validation.get("execution_ids_with_passed_attempts") != (
        observed_passed_ids
    ):
        raise ReportInputError(
            "partial validation passed-execution index drifted"
        )
    passed_by_id = {
        execution_id: rows
        for execution_id, rows in all_passed_by_id.items()
        if (
            method_family is None
            or (
                "append"
                if _method_key(
                    str(expected_jobs[execution_id]["route_id"])
                )
                == "append"
                else "ra"
            )
            == method_family
        )
    }
    if not passed_by_id:
        raise ReportInputError(
            "partial-progress mode requires at least one matching passed "
            "attempt"
        )
    if method_family is None and len(passed_by_id) == 48:
        raise ReportInputError(
            "all 48 cells passed; final mode requires explicit selection"
        )
    ambiguous = sorted(
        execution_id
        for execution_id, rows in passed_by_id.items()
        if len(rows) != 1
    )
    if ambiguous:
        raise ReportInputError(
            "partial-progress mode will not choose among passed retries: "
            + ", ".join(ambiguous)
        )

    cells: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    included_execution_ids = sorted(passed_by_id)
    for execution_id in included_execution_ids:
        attempt_row = passed_by_id[execution_id][0]
        cell, source_row = _load_validated_attempt_cell(
            execution_id=execution_id,
            expected_job=expected_jobs[execution_id],
            attempt_relative_path=str(attempt_row.get("path", "")),
            expected_attempt_sha256=str(attempt_row.get("sha256", "")),
            expected_worker_receipt_sha256=str(
                attempt_row.get("worker_receipt_sha256", "")
            ),
            fetched_dir=fetched_dir,
            terminal_qiskit_compiler=terminal_qiskit_compiler,
        )
        cells.append(cell)
        source_rows.append(source_row)
    return cells, {
        "validation": {
            "path": str(validation_path),
            "sha256": validation_sha,
            "file_sha256": _sha256_file(validation_path),
        },
        "inclusion_policy": (
            "all_execution_ids_with_exactly_one_passed_validated_attempt_v1"
        ),
        "method_family": "all" if method_family is None else method_family,
        "automatic_attempt_selection_performed": False,
        "excluded_nonpassed_execution_ids": sorted(
            excluded_nonpassed_execution_ids
        ),
        "included_sources": source_rows,
    }


def _cross_revision_science_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the report-defining science fields without package provenance."""

    return {
        key: value
        for key, value in manifest.items()
        if key != "package_provenance"
    }


def _load_recovery_adapter(
    *,
    adapter_path: Path,
    expected_jobs: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load explicit non-evidentiary recovery rows for the evolving report."""

    resolved = adapter_path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ReportInputError(
            "recovery adapter escapes the active repository"
        ) from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ReportInputError("recovery adapter is unavailable or unsafe")
    payload = _load_object(resolved, label="recovery adapter")
    adapter_digest = _verify_generic_self_digest(
        payload,
        label="recovery adapter",
    )
    if (
        payload.get("schema") != RECOVERY_ADAPTER_SCHEMA
        or payload.get("status") != "passed"
        or payload.get("not_paper_evidence") is not True
    ):
        raise ReportInputError("recovery adapter authority drifted")

    cells: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    recovery_counts = {
        RECOVERY_CROSS_CAMPAIGN_CLASS: 0,
        RECOVERY_G5_UNEXERCISED_CLASS: 0,
    }
    for index, raw in enumerate(
        _sequence(payload.get("cells"), label="recovery adapter cells"),
        start=1,
    ):
        row = _mapping(raw, label=f"recovery adapter row {index}")
        _verify_generic_self_digest(
            row,
            label=f"recovery adapter row {index}",
        )
        target_id = str(row.get("target_execution_id", ""))
        source_id = str(row.get("source_execution_id", ""))
        recovery_class = str(row.get("recovery_class", ""))
        if (
            not target_id
            or not source_id
            or target_id in seen
            or target_id not in expected_jobs
            or recovery_class not in recovery_counts
            or row.get("paper_evidence_eligible") is not False
        ):
            raise ReportInputError(
                f"recovery adapter row {index} identity drifted"
            )
        seen.add(target_id)
        expected_job = expected_jobs[target_id]
        cell = dict(
            _mapping(
                row.get("cell"),
                label=f"recovery adapter row {index} cell",
            )
        )
        if (
            cell.get("execution_id") != target_id
            or cell.get("regime") != expected_job.get("regime_id")
            or cell.get("representation")
            != expected_job.get("candidate_representation")
            or cell.get("method")
            != _method_key(str(expected_job.get("route_id", "")))
        ):
            raise ReportInputError(
                f"{target_id}: recovered cell identity drifted"
            )

        points = [
            dict(_mapping(value, label=f"{target_id} recovered point"))
            for value in _sequence(
                cell.get("points"),
                label=f"{target_id} recovered points",
            )
        ]
        if len(points) != 51:
            raise ReportInputError(
                f"{target_id}: recovered curve is not 0..50"
            )
        for expected_round, point in enumerate(points):
            if (
                _integer(
                    point.get("k"),
                    label=f"{target_id} recovered point k",
                )
                != expected_round
                or _finite(
                    point.get("error"),
                    label=f"{target_id} recovered point error",
                )
                < 0.0
            ):
                raise ReportInputError(
                    f"{target_id}: recovered curve math drifted"
                )
        terminal = _mapping(
            cell.get("terminal"),
            label=f"{target_id} recovered terminal",
        )
        expected_status = (
            "complete-Xrev"
            if recovery_class == RECOVERY_CROSS_CAMPAIGN_CLASS
            else "complete-G5*"
        )
        if (
            terminal.get("status") != expected_status
            or terminal.get("k") != 50
            or not math.isclose(
                _finite(
                    terminal.get("error"),
                    label=f"{target_id} recovered terminal error",
                ),
                float(points[-1]["error"]),
                abs_tol=1.0e-14,
                rel_tol=1.0e-12,
            )
        ):
            raise ReportInputError(
                f"{target_id}: recovered terminal drifted"
            )
        try:
            qiskit_cost_fields({"metrics": dict(terminal)})
        except (TypeError, ValueError) as exc:
            raise ReportInputError(
                f"{target_id}: recovered terminal cost drifted: {exc}"
            ) from exc
        _integer(
            terminal.get("S_alg"),
            label=f"{target_id} recovered terminal S_alg",
        )
        marker = _mapping(
            cell.get("marker"),
            label=f"{target_id} recovered marker",
        )
        marker_k = _integer(
            marker.get("k"),
            label=f"{target_id} recovered marker k",
        )
        if (
            marker_k > 50
            or not math.isclose(
                _finite(
                    marker.get("error"),
                    label=f"{target_id} recovered marker error",
                ),
                float(points[marker_k]["error"]),
                abs_tol=1.0e-14,
                rel_tol=1.0e-12,
            )
        ):
            raise ReportInputError(
                f"{target_id}: recovered marker drifted"
            )
        _finite(
            cell.get("exact_same_cutoff_energy"),
            label=f"{target_id} recovered exact energy",
        )

        source = _mapping(
            row.get("source"),
            label=f"{target_id} recovered source",
        )
        archive = _mapping(
            source.get("archive"),
            label=f"{target_id} recovered archive",
        )
        result = _mapping(
            source.get("result"),
            label=f"{target_id} recovered result",
        )
        for binding, label in (
            (archive, "archive"),
            (result, "result"),
        ):
            if (
                not isinstance(binding.get("sha256"), str)
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(binding.get("sha256")),
                )
                is None
                or _integer(
                    binding.get("size_bytes"),
                    label=f"{target_id} recovered {label} size",
                    minimum=1,
                )
                < 1
            ):
                raise ReportInputError(
                    f"{target_id}: recovered {label} binding drifted"
                )
        package_id = str(source.get("package_id", ""))
        if not package_id:
            raise ReportInputError(
                f"{target_id}: recovered source package is missing"
            )
        qualification = _mapping(
            row.get("qualification"),
            label=f"{target_id} recovered qualification",
        )
        if recovery_class == RECOVERY_CROSS_CAMPAIGN_CLASS:
            if (
                source.get("attempt_status") != "passed"
                or source.get("worker_exit_status") != 0
                or qualification.get("science_equivalence_status")
                != "passed"
                or not target_id.endswith("_always")
            ):
                raise ReportInputError(
                    f"{target_id}: cross-campaign recovery drifted"
                )
        else:
            if (
                source.get("attempt_status")
                != "failed_attempt_retained"
                or source.get("worker_exit_status") != 2
                or qualification.get("route_domain_status")
                != "unexercised"
                or qualification.get("interior_scored_count") != 0
                or qualification.get("full_controller_rounds") != 50
                or qualification.get("execution_manifest_status")
                != "passed"
                or not target_id.endswith("_plateau")
            ):
                raise ReportInputError(
                    f"{target_id}: G5 recovery qualification drifted"
                )

        recovery_counts[recovery_class] += 1
        cell["recovery_class"] = recovery_class
        cell["source_execution_id"] = source_id
        cells.append(cell)
        source_rows.append(
            {
                "execution_id": target_id,
                "source_execution_id": source_id,
                "method_family": "ra",
                "package_id": package_id,
                "recovery_class": recovery_class,
                "paper_evidence_eligible": False,
                "attempt_path": str(archive.get("path", "")),
                "attempt_sha256": str(archive["sha256"]),
                "attempt_size_bytes": int(archive["size_bytes"]),
                "result_file_sha256": str(result["sha256"]),
                "result_size_bytes": int(result["size_bytes"]),
                "terminal": dict(terminal),
                "marker": dict(marker),
                "plotted_point_count": 51,
            }
        )
    if not cells:
        raise ReportInputError("recovery adapter contains no cells")
    declared_counts = _mapping(
        payload.get("recovery_counts"),
        label="recovery adapter counts",
    )
    if any(
        declared_counts.get(key) != value
        for key, value in recovery_counts.items()
    ):
        raise ReportInputError("recovery adapter count closure drifted")
    return cells, {
        "schema": RECOVERY_ADAPTER_SCHEMA,
        "path": str(resolved),
        "sha256": adapter_digest,
        "file_sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        "included_count": len(cells),
        "included_execution_ids": sorted(seen),
        "recovery_counts": recovery_counts,
        "source_package_ids": sorted(
            {str(row["package_id"]) for row in source_rows}
        ),
        "included_sources": source_rows,
        "paper_evidence_eligible": False,
    }


def _load_local_paused_always_prefix(
    *,
    job_path: Path,
    checkpoint_path: Path,
    log_path: Path,
    expected_jobs: Mapping[str, Mapping[str, Any]],
    exact_same_cutoff_energy: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one authenticated, deliberately paused local RA-always prefix."""

    try:
        import ijson
    except ModuleNotFoundError as exc:
        raise ReportInputError(
            "local paused-prefix loading requires ijson"
        ) from exc

    resolved_job = job_path.resolve()
    resolved_checkpoint = checkpoint_path.resolve()
    resolved_log = log_path.resolve()
    for path, label in (
        (resolved_job, "local paused-prefix job"),
        (resolved_checkpoint, "local paused-prefix checkpoint"),
        (resolved_log, "local paused-prefix log"),
    ):
        if not path.is_file() or path.is_symlink():
            raise ReportInputError(f"{label} is unavailable or unsafe")

    job = _load_object(resolved_job, label="local paused-prefix job")
    job_digest = _verify_generic_self_digest(
        job,
        label="local paused-prefix job",
    )
    target_id = str(job.get("base_cell_id", ""))
    source_id = str(job.get("execution_id", ""))
    if target_id not in expected_jobs:
        raise ReportInputError(
            "local paused-prefix target is not in the stationary matrix"
        )
    expected_job = expected_jobs[target_id]
    expected_source_id = (
        f"{target_id}__gradient_stationary__phase1_cost_off"
    )
    if (
        job.get("package_id") != LOCAL_PAUSED_ALWAYS_PACKAGE_ID
        or source_id != expected_source_id
        or job.get("cell_id") != source_id
        or job.get("horizon") != 50
        or job.get("regime_id") != expected_job.get("regime_id")
        or job.get("candidate_representation")
        != expected_job.get("candidate_representation")
        or job.get("route_id") != expected_job.get("route_id")
        or job.get("route_id") != "ra_macro_always"
        or job.get("candidate_representation") != "macro_generator_v1"
        or job.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or job.get("resource_weighting_scope")
        != "late_resource_weighting_v1"
        or job.get("phase1_cost_term") != "disabled_for_phase1_only"
    ):
        raise ReportInputError(
            f"{source_id or target_id}: local paused-prefix axes drifted"
        )

    history_rows: list[dict[str, Any]] = []
    try:
        with resolved_checkpoint.open("rb") as stream:
            for expected_round, raw in enumerate(
                ijson.items(stream, "adapt_vqe.history_tail.item"),
                start=1,
            ):
                row = _mapping(
                    raw,
                    label=f"{source_id} checkpoint round {expected_round}",
                )
                depth = _integer(
                    row.get("depth"),
                    label=f"{source_id} checkpoint depth",
                    minimum=1,
                )
                if depth != expected_round:
                    raise ReportInputError(
                        f"{source_id}: checkpoint rounds are not contiguous"
                    )
                active_prefix = _mapping(
                    row.get("active_prefix_checkpoint"),
                    label=f"{source_id} active prefix {depth}",
                )
                ledger = _mapping(
                    active_prefix.get("estimator_ledger_receipt"),
                    label=f"{source_id} estimator ledger {depth}",
                )
                cumulative = _mapping(
                    ledger.get("cumulative_executed_queries"),
                    label=f"{source_id} cumulative work {depth}",
                )
                if (
                    active_prefix.get("outer_iteration") != depth
                    or active_prefix.get("active_ansatz_depth") != depth
                    or ledger.get("status") != "complete"
                    or ledger.get("outer_iteration") != depth
                ):
                    raise ReportInputError(
                        f"{source_id}: accepted-prefix authentication failed"
                    )
                history_rows.append(
                    {
                        "k": depth,
                        "energy_before": _finite(
                            row.get("energy_before_opt"),
                            label=f"{source_id} energy before {depth}",
                        ),
                        "energy_after": _finite(
                            row.get("energy_after_opt"),
                            label=f"{source_id} energy after {depth}",
                        ),
                        "selected_position": _integer(
                            row.get("selected_position"),
                            label=f"{source_id} selected position {depth}",
                        ),
                        "S_alg": _integer(
                            cumulative.get("S_alg"),
                            label=f"{source_id} S_alg {depth}",
                        ),
                        "checkpoint_sha256": str(
                            active_prefix.get("checkpoint_sha256", "")
                        ),
                    }
                )
    except (OSError, ValueError) as exc:
        raise ReportInputError(
            f"{source_id}: local checkpoint stream failed: {exc}"
        ) from exc
    if not history_rows or len(history_rows) >= 50:
        raise ReportInputError(
            f"{source_id}: paused prefix must contain 1 through 49 rounds"
        )
    previous_after: float | None = None
    for row in history_rows:
        if previous_after is not None and not math.isclose(
            float(row["energy_before"]),
            previous_after,
            abs_tol=1.0e-11,
            rel_tol=1.0e-11,
        ):
            raise ReportInputError(
                f"{source_id}: checkpoint energy continuity drifted"
            )
        previous_after = float(row["energy_after"])

    log_rows: list[tuple[int, int]] = []
    try:
        for line in resolved_log.read_text(encoding="utf-8").splitlines():
            marker = line.find("AI_LOG ")
            if marker < 0:
                continue
            raw = json.loads(line[marker + len("AI_LOG ") :])
            if raw.get("event") == "hardcoded_adapt_iter":
                log_rows.append(
                    (
                        _integer(
                            raw.get("depth"),
                            label=f"{source_id} log depth",
                            minimum=1,
                        ),
                        _integer(
                            raw.get("selected_position"),
                            label=f"{source_id} log selected position",
                        ),
                    )
                )
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportInputError(
            f"{source_id}: local progress log failed: {exc}"
        ) from exc
    if [row[0] for row in log_rows] != list(
        range(1, len(history_rows) + 1)
    ) or any(
        log_position != int(checkpoint_row["selected_position"])
        for (_, log_position), checkpoint_row in zip(
            log_rows,
            history_rows,
            strict=True,
        )
    ):
        raise ReportInputError(
            f"{source_id}: local log/checkpoint round identity drifted"
        )

    points = [
        {
            "k": 0,
            "energy": float(history_rows[0]["energy_before"]),
            "error": abs(
                float(history_rows[0]["energy_before"])
                - exact_same_cutoff_energy
            ),
        }
    ]
    points.extend(
        {
            "k": int(row["k"]),
            "energy": float(row["energy_after"]),
            "error": abs(
                float(row["energy_after"]) - exact_same_cutoff_energy
            ),
        }
        for row in history_rows
    )
    from pipelines.reporting.paper_i_run_summary import (
        PaperIErrorTracePoint,
        select_paper_i_effective_plateau,
    )

    plateau = select_paper_i_effective_plateau(
        tuple(
            PaperIErrorTracePoint(
                controller_round=int(row["k"]),
                absolute_energy_error=float(row["error"]),
            )
            for row in points[1:]
        )
    )
    marker_point = points[int(plateau.controller_round)]
    terminal = points[-1]
    cell = {
        "execution_id": target_id,
        "regime": str(expected_job["regime_id"]),
        "representation": str(expected_job["candidate_representation"]),
        "method": "always",
        "points": points,
        "marker": {
            "k": int(marker_point["k"]),
            "error": float(marker_point["error"]),
            "policy": str(plateau.policy),
        },
        "terminal": {
            "k": int(terminal["k"]),
            "error": float(terminal["error"]),
            "S_alg": int(history_rows[-1]["S_alg"]),
            "status": "paused-local",
        },
        "exact_same_cutoff_energy": exact_same_cutoff_energy,
        "local_paused_prefix": True,
        "source_execution_id": source_id,
    }
    source = {
        "execution_id": target_id,
        "source_execution_id": source_id,
        "method_family": "ra_local_paused_prefix",
        "package_id": str(job["package_id"]),
        "regime_id": str(job["regime_id"]),
        "candidate_representation": str(job["candidate_representation"]),
        "route_id": str(job["route_id"]),
        "plotted_point_count": len(points),
        "paused_controller_round": int(terminal["k"]),
        "terminal": dict(cell["terminal"]),
        "marker": dict(cell["marker"]),
        "job": {
            "path": str(resolved_job),
            "sha256": job_digest,
            "file_sha256": _sha256_file(resolved_job),
            "size_bytes": resolved_job.stat().st_size,
        },
        "checkpoint": {
            "path": str(resolved_checkpoint),
            "sha256": _sha256_file(resolved_checkpoint),
            "size_bytes": resolved_checkpoint.stat().st_size,
            "accepted_prefix_checkpoint_sha256": str(
                history_rows[-1]["checkpoint_sha256"]
            ),
        },
        "log": {
            "path": str(resolved_log),
            "sha256": _sha256_file(resolved_log),
            "size_bytes": resolved_log.stat().st_size,
        },
        "paper_evidence_eligible": False,
        "continuation_state": "preserved_checkpoint_requires_route_exact_adapter",
    }
    return cell, source


def _load_global_singleton_weak_weak_diagnostic(
    adapter_path: Path,
) -> dict[str, Any]:
    """Load the authenticated, diagnostic-only weak--weak insertion pair."""

    resolved = adapter_path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ReportInputError(
            "global-singleton diagnostic adapter escapes the active repository"
        ) from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ReportInputError(
            "global-singleton diagnostic adapter is unavailable or unsafe"
        )
    payload = _load_object(
        resolved,
        label="global-singleton weak-weak diagnostic adapter",
    )
    adapter_digest = _verify_generic_self_digest(
        payload,
        label="global-singleton weak-weak diagnostic adapter",
    )
    if (
        payload.get("schema") != GLOBAL_SINGLETON_WW_DIAGNOSTIC_SCHEMA
        or payload.get("status") != "passed"
        or payload.get("diagnostic_only") is not True
        or payload.get("paper_evidence_adopted") is not False
        or payload.get("campaign_id") != GLOBAL_SINGLETON_WW_CAMPAIGN_ID
        or payload.get("regime_id") != "weak_weak"
        or payload.get("nph") != 3
        or payload.get("horizon") != 50
    ):
        raise ReportInputError(
            "global-singleton weak-weak diagnostic authority drifted"
        )
    cross_arm = _mapping(
        payload.get("cross_arm_audit"),
        label="global-singleton weak-weak cross-arm audit",
    )
    if (
        cross_arm.get("status") != "passed"
        or cross_arm.get("allowed_axis") != "insertion_policy"
        or cross_arm.get("canonical_sha256")
        != GLOBAL_SINGLETON_WW_CROSS_ARM_SHA256
    ):
        raise ReportInputError(
            "global-singleton weak-weak cross-arm equality drifted"
        )

    arms: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(
        _sequence(
            payload.get("arms"),
            label="global-singleton weak-weak diagnostic arms",
        ),
        start=1,
    ):
        arm = dict(
            _mapping(
                raw,
                label=f"global-singleton weak-weak arm {index}",
            )
        )
        _verify_generic_self_digest(
            arm,
            label=f"global-singleton weak-weak arm {index}",
        )
        policy = str(arm.get("insertion_policy", ""))
        if policy not in GLOBAL_SINGLETON_WW_POLICIES or policy in arms:
            raise ReportInputError(
                "global-singleton weak-weak insertion-policy matrix drifted"
            )
        expected_suffix = {
            "append_commutation_reduced": (
                "__ra_global_singleton_append_commutation_reduced"
            ),
            "plateau_commutation": (
                "__ra_global_singleton_plateau_commutation"
            ),
        }[policy]
        expected_route = {
            "append_commutation_reduced": (
                "ra_global_singleton_append_commutation_reduced"
            ),
            "plateau_commutation": (
                "ra_global_singleton_plateau_commutation"
            ),
        }[policy]
        execution_id = str(arm.get("execution_id", ""))
        if (
            arm.get("schema")
            != (
                "paper_i_ra_adapt_global_singleton_weak_weak_"
                "comparison_arm_v1"
            )
            or not execution_id.startswith(
                "global_singleton__weak_weak__nph3"
            )
            or not execution_id.endswith(expected_suffix)
            or arm.get("route_id") != expected_route
        ):
            raise ReportInputError(
                f"global-singleton {policy} execution identity drifted"
            )
        points = [
            dict(
                _mapping(
                    value,
                    label=f"global-singleton {policy} plotted point",
                )
            )
            for value in _sequence(
                arm.get("points"),
                label=f"global-singleton {policy} plotted points",
            )
        ]
        if len(points) != 51:
            raise ReportInputError(
                f"global-singleton {policy} curve is not 0..50"
            )
        for expected_round, point in enumerate(points):
            if (
                _integer(
                    point.get("k"),
                    label=f"global-singleton {policy} point k",
                )
                != expected_round
                or _finite(
                    point.get("error"),
                    label=f"global-singleton {policy} point error",
                )
                < 0.0
            ):
                raise ReportInputError(
                    f"global-singleton {policy} curve math drifted"
                )
        terminal = _mapping(
            arm.get("terminal"),
            label=f"global-singleton {policy} terminal",
        )
        effective = _mapping(
            arm.get("effective_plateau"),
            label=f"global-singleton {policy} effective plateau",
        )
        if (
            _integer(
                terminal.get("k"),
                label=f"global-singleton {policy} terminal k",
            )
            != 50
            or not math.isclose(
                _finite(
                    terminal.get("error"),
                    label=f"global-singleton {policy} terminal error",
                ),
                float(points[-1]["error"]),
                abs_tol=1.0e-16,
                rel_tol=1.0e-12,
            )
        ):
            raise ReportInputError(
                f"global-singleton {policy} terminal drifted"
            )
        effective_k = _integer(
            effective.get("k"),
            label=f"global-singleton {policy} effective plateau k",
        )
        if (
            effective_k > 50
            or not math.isclose(
                _finite(
                    effective.get("error"),
                    label=f"global-singleton {policy} effective error",
                ),
                float(points[effective_k]["error"]),
                abs_tol=1.0e-16,
                rel_tol=1.0e-12,
            )
        ):
            raise ReportInputError(
                f"global-singleton {policy} effective plateau drifted"
            )
        for observation_name, observation in (
            ("terminal", terminal),
            ("effective plateau", effective),
        ):
            for field in ("S_alg", "N2q", "D2q", "Dc", "W1q", "B1q"):
                _integer(
                    observation.get(field),
                    label=(
                        f"global-singleton {policy} "
                        f"{observation_name} {field}"
                    ),
                )
            if not isinstance(observation.get("compile_convention"), str):
                raise ReportInputError(
                    f"global-singleton {policy} compile convention drifted"
                )
        insertion = _mapping(
            arm.get("insertion_counts"),
            label=f"global-singleton {policy} insertion counts",
        )
        round_count = _integer(
            insertion.get("round_count"),
            label=f"global-singleton {policy} insertion round count",
        )
        append_count = _integer(
            insertion.get("append_count"),
            label=f"global-singleton {policy} append count",
        )
        interior_count = _integer(
            insertion.get("interior_count"),
            label=f"global-singleton {policy} interior count",
        )
        if (
            round_count != 50
            or append_count + interior_count != round_count
            or (
                policy == "append_commutation_reduced"
                and interior_count != 0
            )
        ):
            raise ReportInputError(
                f"global-singleton {policy} insertion count drifted"
            )
        if (
            policy == "plateau_commutation"
            and (
                interior_count < 1
                or _integer(
                    insertion.get("first_interior_round"),
                    label="global-singleton plateau first interior round",
                    minimum=1,
                )
                > 50
            )
        ):
            raise ReportInputError(
                "global-singleton plateau insertion domain was unexercised"
            )
        qualification = _mapping(
            arm.get("qualification"),
            label=f"global-singleton {policy} qualification",
        )
        route_profile = str(qualification.get("route_profile", ""))
        if (
            qualification.get("status") != "passed"
            or qualification.get("result_schema")
            != "paper_i_ra_adapt_result_v1"
            or qualification.get("full_controller_rounds") != 50
            or qualification.get("same_cutoff_trace_math") != "passed"
            or qualification.get("canonical_work_closure") != "passed"
            or qualification.get("authenticated_prefix_reconstruction")
            != "passed"
            or qualification.get("serialized_plateau_qiskit_cross_check")
            != "passed"
            or "__stationary_source_response_v1__" not in route_profile
            or not route_profile.endswith(
                "__all_phase_resource_weighting_v1"
            )
        ):
            raise ReportInputError(
                f"global-singleton {policy} qualification drifted"
            )
        _finite(
            qualification.get("exact_same_cutoff_energy"),
            label=f"global-singleton {policy} exact energy",
        )
        source = _mapping(
            arm.get("source"),
            label=f"global-singleton {policy} source",
        )
        archive = _mapping(
            source.get("archive"),
            label=f"global-singleton {policy} archive",
        )
        if (
            not isinstance(archive.get("path"), str)
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(archive.get("sha256", "")),
            )
            is None
            or _integer(
                archive.get("size_bytes"),
                label=f"global-singleton {policy} archive size",
                minimum=1,
            )
            < 1
        ):
            raise ReportInputError(
                f"global-singleton {policy} archive binding drifted"
            )
        arms[policy] = {
            **arm,
            "points": points,
            "terminal": dict(terminal),
            "effective_plateau": dict(effective),
            "insertion_counts": dict(insertion),
            "qualification": dict(qualification),
        }
    if tuple(arms) != GLOBAL_SINGLETON_WW_POLICIES:
        raise ReportInputError(
            "global-singleton weak-weak arm order or coverage drifted"
        )
    exact_energies = {
        float(
            _mapping(
                arm.get("qualification"),
                label="global-singleton weak-weak qualification",
            )["exact_same_cutoff_energy"]
        )
        for arm in arms.values()
    }
    if len(exact_energies) != 1:
        raise ReportInputError(
            "global-singleton weak-weak exact-energy binding drifted"
        )

    comparison = dict(
        _mapping(
            payload.get("comparison"),
            label="global-singleton weak-weak comparison",
        )
    )
    if (
        comparison.get("comparison_order")
        != list(GLOBAL_SINGLETON_WW_POLICIES)
        or not math.isclose(
            _finite(
                comparison.get("same_cutoff_exact_energy"),
                label="global-singleton comparison exact energy",
            ),
            next(iter(exact_energies)),
            abs_tol=0.0,
            rel_tol=0.0,
        )
    ):
        raise ReportInputError(
            "global-singleton weak-weak comparison closure drifted"
        )
    append_s_alg = int(
        _mapping(
            arms["append_commutation_reduced"]["terminal"],
            label="global-singleton append terminal",
        )["S_alg"]
    )
    plateau_s_alg = int(
        _mapping(
            arms["plateau_commutation"]["terminal"],
            label="global-singleton plateau terminal",
        )["S_alg"]
    )
    if append_s_alg < 1 or plateau_s_alg < 1:
        raise ReportInputError(
            "global-singleton weak-weak work comparison drifted"
        )
    return {
        **payload,
        "arms_by_policy": arms,
        "comparison": comparison,
        "adapter_source": {
            "path": str(resolved),
            "sha256": adapter_digest,
            "file_sha256": _sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        },
        "derived": {
            "terminal_s_alg_ratio_plateau_over_append": (
                plateau_s_alg / append_s_alg
            ),
            "terminal_error_difference_plateau_minus_append": (
                float(
                    _mapping(
                        arms["plateau_commutation"]["terminal"],
                        label="global-singleton plateau terminal",
                    )["error"]
                )
                - float(
                    _mapping(
                        arms["append_commutation_reduced"]["terminal"],
                        label="global-singleton append terminal",
                    )["error"]
                )
            ),
        },
    }


def load_cross_revision_partial_cells(
    *,
    source_specs: Sequence[Mapping[str, Any]],
    recovery_adapter_paths: Sequence[Path] = (),
    terminal_qiskit_compiler: Callable[[Any], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load disjoint method families from explicit validated package sources."""

    if not source_specs:
        raise ReportInputError(
            "cross-revision progress requires explicit partial sources"
        )
    included_cells: list[dict[str, Any]] = []
    included_sources: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    seen_execution_ids: set[str] = set()
    observed_families: set[str] = set()
    observed_package_ids: set[str] = set()
    science_manifest: dict[str, Any] | None = None
    package_cache: dict[Path, tuple[dict[str, Any], dict[str, Any]]] = {}
    anchor_package_dir: Path | None = None

    for source_index, raw_spec in enumerate(source_specs, start=1):
        spec = _mapping(raw_spec, label=f"partial source {source_index}")
        method_family = str(spec.get("method_family", ""))
        if method_family not in {"append", "ra"}:
            raise ReportInputError(
                f"partial source {source_index} has unknown method family"
            )
        package_dir = Path(str(spec.get("package_dir", ""))).resolve()
        validation_path = Path(
            str(spec.get("validation_path", ""))
        ).resolve()
        fetched_dir = Path(str(spec.get("fetched_dir", ""))).resolve()
        package_id = _configure_package_dir(package_dir)
        anchor_package_dir = package_dir
        observed_families.add(method_family)
        observed_package_ids.add(package_id)

        cached = package_cache.get(package_dir)
        if cached is None:
            parameter_manifest = _parameter_manifest()
            package_sources = _package_sources()
            cached = (parameter_manifest, package_sources)
            package_cache[package_dir] = cached
        else:
            parameter_manifest, package_sources = cached
        current_science = _cross_revision_science_manifest(
            parameter_manifest
        )
        if science_manifest is None:
            science_manifest = current_science
        elif current_science != science_manifest:
            raise ReportInputError(
                "cross-revision package science manifests disagree"
            )

        cells, validation_sources = load_partial_cells(
            validation_path=validation_path,
            fetched_dir=fetched_dir,
            terminal_qiskit_compiler=terminal_qiskit_compiler,
            method_family=method_family,
        )
        source_execution_ids = sorted(
            str(cell["execution_id"]) for cell in cells
        )
        overlap = sorted(
            set(source_execution_ids).intersection(seen_execution_ids)
        )
        if overlap:
            raise ReportInputError(
                "cross-revision sources overlap successful cells: "
                + ", ".join(overlap)
            )
        seen_execution_ids.update(source_execution_ids)
        included_cells.extend(cells)

        source_rows = []
        for row in _sequence(
            validation_sources.get("included_sources"),
            label=f"partial source {source_index} included sources",
        ):
            source_row = dict(
                _mapping(
                    row,
                    label=(
                        f"partial source {source_index} included source"
                    ),
                )
            )
            source_row.update(
                {
                    "method_family": method_family,
                    "package_id": package_id,
                    "core_materialization_id": (
                        package_sources["core_materialization_id"]
                    ),
                    "source_receipt_index": source_index,
                }
            )
            included_sources.append(source_row)
            source_rows.append(source_row)

        validation_binding = dict(
            _mapping(
                validation_sources.get("validation"),
                label=f"partial source {source_index} validation binding",
            )
        )
        source_records.append(
            {
                "source_receipt_index": source_index,
                "method_family": method_family,
                "package_id": package_id,
                "package_dir": str(package_dir),
                "core_materialization_id": package_sources[
                    "core_materialization_id"
                ],
                "package_sources": package_sources,
                "validation": validation_binding,
                "fetched_dir": str(fetched_dir),
                "included_execution_ids": source_execution_ids,
                "included_count": len(source_execution_ids),
                "excluded_nonpassed_execution_ids": list(
                    validation_sources.get(
                        "excluded_nonpassed_execution_ids", ()
                    )
                ),
                "inclusion_policy": validation_sources[
                    "inclusion_policy"
                ],
                "automatic_attempt_selection_performed": False,
            }
        )

    if observed_families != {"append", "ra"}:
        raise ReportInputError(
            "cross-revision progress requires both append and RA sources"
        )
    if len(observed_package_ids) < 2:
        raise ReportInputError(
            "cross-revision progress requires at least two package identities"
        )
    if science_manifest is None or anchor_package_dir is None:
        raise ReportInputError("cross-revision source resolution failed")

    _configure_package_dir(anchor_package_dir)
    recovery_records: list[dict[str, Any]] = []
    if recovery_adapter_paths:
        expected_jobs = _expected_jobs()
        for adapter_path in recovery_adapter_paths:
            recovered_cells, recovery_record = _load_recovery_adapter(
                adapter_path=adapter_path,
                expected_jobs=expected_jobs,
            )
            recovered_ids = sorted(
                str(cell["execution_id"]) for cell in recovered_cells
            )
            overlap = sorted(set(recovered_ids).intersection(seen_execution_ids))
            if overlap:
                raise ReportInputError(
                    "recovery adapter overlaps an included cell: "
                    + ", ".join(overlap)
                )
            seen_execution_ids.update(recovered_ids)
            included_cells.extend(recovered_cells)
            source_index = len(source_records) + 1
            record = {
                **recovery_record,
                "source_receipt_index": source_index,
                "method_family": "ra_recovery_adapter",
                "inclusion_policy": (
                    "explicit_non_evidentiary_recovery_adapter_v1"
                ),
                "automatic_attempt_selection_performed": False,
            }
            recovery_records.append(record)
            source_records.append(record)
            observed_package_ids.update(
                str(value)
                for value in _sequence(
                    recovery_record.get("source_package_ids"),
                    label="recovery source package ids",
                )
            )
            for raw_source in _sequence(
                recovery_record.get("included_sources"),
                label="recovery included sources",
            ):
                included_sources.append(
                    {
                        **dict(
                            _mapping(
                                raw_source,
                                label="recovery included source",
                            )
                        ),
                        "source_receipt_index": source_index,
                    }
                )

    cross_package_provenance = {
        "cross_revision": True,
        "source_package_count": len(observed_package_ids),
        "source_receipt_count": len(source_records),
        "package_ids": sorted(observed_package_ids),
        "sources": [
            {
                "source_receipt_index": row["source_receipt_index"],
                "method_family": row["method_family"],
                "package_id": row["package_id"],
                "core_materialization_id": row[
                    "core_materialization_id"
                ],
                "package_manifest_sha256": row["package_sources"][
                    "package_manifest"
                ]["sha256"],
                "source_archive_sha256": row["package_sources"][
                    "source_archive_sha256"
                ],
                "validation_sha256": row["validation"]["sha256"],
            }
            for row in source_records
            if "package_sources" in row
        ],
        "recovery_adapter_count": len(recovery_records),
        "recovery_cell_count": sum(
            int(row["included_count"]) for row in recovery_records
        ),
        "recovery_adapters": [
            {
                "source_receipt_index": row["source_receipt_index"],
                "schema": row["schema"],
                "sha256": row["sha256"],
                "file_sha256": row["file_sha256"],
                "included_count": row["included_count"],
                "included_execution_ids": row["included_execution_ids"],
                "recovery_counts": row["recovery_counts"],
                "source_package_ids": row["source_package_ids"],
                "paper_evidence_eligible": False,
            }
            for row in recovery_records
        ],
    }
    parameter_manifest = {
        **science_manifest,
        "package_provenance": cross_package_provenance,
    }
    return included_cells, {
        "source_policy": (
            "append_from_explicit_append_sources_ra_from_explicit_ra_"
            "sources_no_cross_source_retry_selection"
            + (
                "_plus_explicit_non_evidentiary_recovery_adapters_v1"
                if recovery_records
                else "_v1"
            )
        ),
        "automatic_attempt_selection_performed": False,
        "source_records": source_records,
        "included_sources": included_sources,
    }, parameter_manifest


def _pending_cells() -> list[dict[str, Any]]:
    jobs = _expected_jobs()
    return [
        {
            "execution_id": execution_id,
            "regime": str(job["regime_id"]),
            "representation": str(job["candidate_representation"]),
            "method": _method_key(str(job["route_id"])),
            "points": [],
            "marker": {
                "k": None,
                "error": None,
                "policy": "pending",
            },
            "terminal": {
                "k": None,
                "error": None,
                "N2q": None,
                "D2q": None,
                "Dc": None,
                "W1q": None,
                "B1q": None,
                "qiskit_basis_work_status": "pending",
                "qiskit_basis_work_schema": None,
                "S_alg": None,
                "status": "pending",
            },
        }
        for execution_id, job in sorted(jobs.items())
    ]


def _merge_partial_with_pending(
    included_cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pending = _pending_cells()
    merged = {
        (
            str(row["representation"]),
            str(row["regime"]),
            str(row["method"]),
        ): dict(row)
        for row in pending
    }
    observed: set[tuple[str, str, str]] = set()
    for raw in included_cells:
        row = _mapping(raw, label="partial included cell")
        key = (
            str(row.get("representation", "")),
            str(row.get("regime", "")),
            str(row.get("method", "")),
        )
        if key not in merged or key in observed:
            raise ReportInputError(
                f"partial cell is outside/duplicated in the 48-cell matrix: "
                f"{key}"
            )
        if row.get("execution_id") != merged[key]["execution_id"]:
            raise ReportInputError(
                f"partial cell execution identity drifted: {key}"
            )
        merged[key] = dict(row)
        observed.add(key)
    ordered = [
        merged[
            (
                str(row["representation"]),
                str(row["regime"]),
                str(row["method"]),
            )
        ]
        for row in pending
    ]
    _cell_index(ordered)
    return ordered


def _cell_index(
    cells: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    indexed = {
        (
            str(row["representation"]),
            str(row["regime"]),
            str(row["method"]),
        ): row
        for row in cells
    }
    if len(indexed) != 48:
        raise ReportInputError("report cells do not form 48 unique identities")
    return indexed


def _format_error(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2e}"


def _format_integer(value: Any) -> str:
    return "--" if value is None else f"{int(value):,}"


def _format_s_alg_mathtext(value: int) -> str:
    """Format estimator work for compact Matplotlib mathtext tuples."""

    mantissa, exponent = f"{int(value):.1e}".split("e")
    return rf"{mantissa}\mathrm{{e}}{int(exponent)}"


def _terminal_cost_plot_overlay(
    *,
    terminal: Mapping[str, Any],
    method: str,
) -> str | None:
    """Return the canonical round-50 tuple annotation for a completed curve."""

    if (
        not str(terminal.get("status", "")).startswith("complete")
        or terminal.get("k") != 50
    ):
        return None
    try:
        marker = PLOT_TUPLE_MARKERS[method]
        return paper_i_cost_tuple_latex(
            terminal,
            marker=marker,
            format_s_alg=_format_s_alg_mathtext,
        )
    except KeyError as exc:
        raise ReportInputError(f"unknown tuple-overlay method: {method}") from exc
    except (TypeError, ValueError) as exc:
        raise ReportInputError(
            "completed round-50 plot tuple is incomplete: "
            f"method={method}: {exc}"
        ) from exc


def _point_error_at(
    cell: Mapping[str, Any],
    *,
    controller_round: int,
    label: str,
) -> float:
    matches = [
        _mapping(row, label=f"{label} point")
        for row in _sequence(cell.get("points"), label=f"{label} points")
        if _integer(
            _mapping(row, label=f"{label} candidate point").get("k"),
            label=f"{label} candidate round",
        )
        == controller_round
    ]
    if len(matches) != 1:
        raise ReportInputError(
            f"{label} must contain exactly one round-{controller_round} point"
        )
    return _finite(matches[0].get("error"), label=f"{label} error")


def _load_cumulative_plateau_macro_diagnostic(
    *,
    run_dir: Path,
    plateau_cell: Mapping[str, Any],
    append_cell: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the completed cumulative-relative plateau prefix."""

    resolved = run_dir.resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ReportInputError(
            "cumulative-plateau macro diagnostic directory is unavailable"
        )
    paths = {
        name: resolved / f"{name}.json"
        for name in (
            "run_manifest",
            "execution_authorization",
            "terminal_receipt",
            "result",
            "checkpoint",
            "estimator_ledger",
        )
    }
    manifest, manifest_digest = _verified_object(
        paths["run_manifest"],
        label="cumulative-plateau macro run manifest",
    )
    authorization, authorization_digest = _verified_object(
        paths["execution_authorization"],
        label="cumulative-plateau macro execution authorization",
    )
    receipt, receipt_digest = _verified_object(
        paths["terminal_receipt"],
        label="cumulative-plateau macro terminal receipt",
    )
    for path in paths.values():
        if not path.is_file() or path.is_symlink():
            raise ReportInputError(
                f"cumulative-plateau macro source is unavailable: {path}"
            )
    result = _load_object(
        paths["result"],
        label="cumulative-plateau macro result",
    )
    if (
        manifest.get("schema") != "paper_i_ra_adapt_local_run_manifest_v1"
        or manifest.get("cell_id") != CUMULATIVE_PLATEAU_MACRO_EXECUTION_ID
        or manifest.get("candidate_representation")
        != "macro_generator_v1"
        or manifest.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or manifest.get("resource_weighting_scope")
        != "late_resource_weighting_v1"
        or manifest.get("optimizer") != "powell"
        or manifest.get("optimizer_maxiter") != 200
        or manifest.get("adapt_seed") != 7
        or manifest.get("protocol_horizon") != 50
        or manifest.get("operational_maximum_controller_rounds") != 20
        or not math.isclose(
            _finite(
                manifest.get("plateau_cumulative_decrease_ratio_threshold"),
                label="cumulative-plateau ratio threshold",
            ),
            1.0e-4,
            abs_tol=0.0,
            rel_tol=0.0,
        )
        or authorization.get("schema")
        != "paper_i_ra_adapt_local_execution_authorization_v1"
        or authorization.get("cell_id")
        != CUMULATIVE_PLATEAU_MACRO_EXECUTION_ID
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not False
        or authorization.get("maximum_controller_rounds") != 20
        or manifest.get("execution_authorization_sha256")
        != authorization_digest
        or receipt.get("schema")
        != "paper_i_ra_adapt_local_terminal_receipt_v1"
        or receipt.get("status") != "passed"
        or receipt.get("cell_id") != CUMULATIVE_PLATEAU_MACRO_EXECUTION_ID
        or receipt.get("accepted_controller_rounds") != 20
        or receipt.get("manifest_sha256") != manifest_digest
        or receipt.get("protocol_sha256") != manifest.get("protocol_sha256")
        or authorization.get("protocol_sha256")
        != manifest.get("protocol_sha256")
        or receipt.get("result_sha256") != _sha256_file(paths["result"])
        or receipt.get("checkpoint_sha256")
        != _sha256_file(paths["checkpoint"])
        or receipt.get("estimator_ledger_sha256")
        != _sha256_file(paths["estimator_ledger"])
    ):
        raise ReportInputError(
            "cumulative-plateau macro identity or terminal binding drifted"
        )
    embedded_protocol = _mapping(
        result.get("protocol"),
        label="cumulative-plateau embedded protocol",
    )
    run = _mapping(result.get("run"), label="cumulative-plateau result run")
    route = _mapping(run.get("route"), label="cumulative-plateau result route")
    stop = _mapping(run.get("stop"), label="cumulative-plateau stop receipt")
    policy = _mapping(
        result.get("policy"),
        label="cumulative-plateau result policy",
    )
    transitions = [
        _mapping(row, label="cumulative-plateau accepted transition")
        for row in _sequence(
            run.get("accepted_transitions"),
            label="cumulative-plateau accepted transitions",
        )
    ]
    summary = _mapping(
        run.get("paper_i_summary"),
        label="cumulative-plateau embedded summary",
    )
    trace = [
        _mapping(row, label="cumulative-plateau error trace row")
        for row in _sequence(
            summary.get("accepted_error_trace"),
            label="cumulative-plateau error trace",
        )
    ]
    if (
        result.get("schema") != "paper_i_ra_adapt_result_v1"
        or embedded_protocol.get("sha256") != manifest.get("protocol_sha256")
        or embedded_protocol.get("candidate_representation")
        != "macro_generator_v1"
        or embedded_protocol.get("horizon") != 50
        or embedded_protocol.get("optimizer") != "powell"
        or embedded_protocol.get("optimizer_maxiter") != 200
        or route.get("insertion_policy") != "plateau_commutation"
        or policy.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or policy.get("resource_weighting_scope")
        != "late_resource_weighting_v1"
        or stop.get("completed_controller_rounds") != 20
        or stop.get("primary_reason") != "maximum_controller_rounds"
        or len(transitions) != 20
        or len(trace) != 20
        or [row.get("controller_round") for row in transitions]
        != list(range(1, 21))
        or [row.get("controller_round") for row in trace]
        != list(range(1, 21))
    ):
        raise ReportInputError(
            "cumulative-plateau macro scientific projection drifted"
        )
    exact_energy = _finite(
        manifest.get("exact_same_cutoff_energy"),
        label="cumulative-plateau exact same-cutoff energy",
    )
    for baseline_label, cell in (
        ("stationary RA plateau", plateau_cell),
        ("Append", append_cell),
    ):
        if not math.isclose(
            exact_energy,
            _finite(
                cell.get("exact_same_cutoff_energy"),
                label=f"{baseline_label} exact same-cutoff energy",
            ),
            abs_tol=1.0e-12,
            rel_tol=0.0,
        ):
            raise ReportInputError(
                f"cumulative-plateau exact reference disagrees with {baseline_label}"
            )
    initial_energy = _finite(
        transitions[0].get("energy_before"),
        label="cumulative-plateau initial energy",
    )
    points = [{"k": 0, "error": abs(initial_energy - exact_energy)}]
    for transition, row in zip(transitions, trace, strict=True):
        controller_round = _integer(
            row.get("controller_round"),
            label="cumulative-plateau trace round",
            minimum=1,
        )
        energy = _finite(
            row.get("accepted_energy"),
            label="cumulative-plateau accepted energy",
        )
        error = _finite(
            row.get("absolute_energy_error"),
            label="cumulative-plateau accepted error",
        )
        if (
            not math.isclose(
                energy,
                _finite(
                    transition.get("energy_after"),
                    label="cumulative-plateau transition energy",
                ),
                abs_tol=1.0e-12,
                rel_tol=1.0e-11,
            )
            or not math.isclose(
                error,
                abs(energy - exact_energy),
                abs_tol=1.0e-12,
                rel_tol=1.0e-11,
            )
        ):
            raise ReportInputError(
                "cumulative-plateau accepted trajectory disagrees with its error trace"
            )
        points.append({"k": controller_round, "error": error})
    final_error = float(points[-1]["error"])
    if not math.isclose(
        final_error,
        _finite(
            receipt.get("final_same_cutoff_delta_e"),
            label="cumulative-plateau terminal error",
        ),
        abs_tol=1.0e-12,
        rel_tol=1.0e-11,
    ):
        raise ReportInputError(
            "cumulative-plateau terminal error disagrees with its receipt"
        )
    interior_rounds = [
        _integer(
            row.get("controller_round"),
            label="cumulative-plateau interior round",
        )
        for row in transitions
        if 0
        < _integer(
            row.get("insertion_position"),
            label="cumulative-plateau insertion position",
        )
        < _integer(
            row.get("controller_round"),
            label="cumulative-plateau insertion round",
        )
        - 1
    ]
    plateau_k20 = _point_error_at(
        plateau_cell,
        controller_round=20,
        label="matched stationary RA plateau",
    )
    append_k20 = _point_error_at(
        append_cell,
        controller_round=20,
        label="matched Append",
    )
    plateau_k50 = _point_error_at(
        plateau_cell,
        controller_round=50,
        label="matched stationary RA plateau",
    )
    append_k50 = _point_error_at(
        append_cell,
        controller_round=50,
        label="matched Append",
    )
    return {
        "schema": "paper_i_ra_adapt_cumulative_plateau_macro_diagnostic_v1",
        "status": "completed_20_round_diagnostic",
        "not_paper_evidence": True,
        "execution_id": CUMULATIVE_PLATEAU_MACRO_EXECUTION_ID,
        "regime": "intermediate_strong",
        "representation": "macro_generator_v1",
        "points": points,
        "marker": {
            "k": 20,
            "error": final_error,
            "policy": "diagnostic_terminal_observed_point",
        },
        "interior_selection_rounds": interior_rounds,
        "first_interior_selection_round": (
            None if not interior_rounds else interior_rounds[0]
        ),
        "comparison": {
            "controller_round": 20,
            "cumulative_relative_plateau_delta_e": final_error,
            "stationary_ra_absolute_plateau_delta_e": plateau_k20,
            "append_adapt_delta_e": append_k20,
            "stationary_ra_absolute_plateau_over_cumulative_ratio": (
                plateau_k20 / max(final_error, 1.0e-300)
            ),
            "append_adapt_over_cumulative_ratio": (
                append_k20 / max(final_error, 1.0e-300)
            ),
            "stationary_ra_absolute_plateau_round50_delta_e": plateau_k50,
            "append_adapt_round50_delta_e": append_k50,
            "stationary_ra_round50_over_cumulative_round20_ratio": (
                plateau_k50 / max(final_error, 1.0e-300)
            ),
            "append_round50_over_cumulative_round20_ratio": (
                append_k50 / max(final_error, 1.0e-300)
            ),
        },
        "source_bindings": {
            "run_manifest": {
                "path": str(paths["run_manifest"]),
                "canonical_sha256": manifest_digest,
                "file_sha256": _sha256_file(paths["run_manifest"]),
            },
            "execution_authorization": {
                "path": str(paths["execution_authorization"]),
                "canonical_sha256": authorization_digest,
                "file_sha256": _sha256_file(paths["execution_authorization"]),
            },
            "terminal_receipt": {
                "path": str(paths["terminal_receipt"]),
                "canonical_sha256": receipt_digest,
                "file_sha256": _sha256_file(paths["terminal_receipt"]),
            },
            "result": {
                "path": str(paths["result"]),
                "sha256": _sha256_file(paths["result"]),
                "size_bytes": paths["result"].stat().st_size,
            },
            "checkpoint": {
                "path": str(paths["checkpoint"]),
                "sha256": _sha256_file(paths["checkpoint"]),
                "size_bytes": paths["checkpoint"].stat().st_size,
            },
            "estimator_ledger": {
                "path": str(paths["estimator_ledger"]),
                "sha256": _sha256_file(paths["estimator_ledger"]),
                "size_bytes": paths["estimator_ledger"].stat().st_size,
            },
        },
    }


def _render_plot_grid(
    *,
    cells: Sequence[Mapping[str, Any]],
    representation_key: str,
    vector_output: Path,
    png_output: Path,
    pending: bool,
    partial: bool = False,
    diagnostic_overlays: Sequence[Mapping[str, Any]] = (),
) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

    if pending and partial:
        raise ReportInputError("plot grid cannot be both pending and partial")
    representation = REPRESENTATIONS[representation_key]
    indexed = _cell_index(cells)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.75, 4.45), dpi=240)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.072,
        right=0.992,
        top=0.805,
        bottom=0.115,
        wspace=0.22,
        hspace=0.34,
    )
    for index, regime in enumerate(REGIME_ORDER):
        ax = fig.add_subplot(grid[index // 3, index % 3])
        observed_errors: list[float] = []
        missing_methods: list[str] = []
        tuple_overlays: list[tuple[str, str]] = []
        for method in METHOD_ORDER:
            cell = indexed[(representation, regime, method)]
            points = list(cell["points"])
            if not points:
                missing_methods.append(method)
                continue
            x = [int(row["k"]) for row in points]
            raw_y = [float(row["error"]) for row in points]
            y = [max(value, 1.0e-16) for value in raw_y]
            style = METHODS[method]
            ax.plot(
                x,
                y,
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle="-",
            )
            marker = _mapping(
                cell.get("marker"),
                label=f"{cell['execution_id']} marker",
            )
            marker_k = _integer(
                marker.get("k"),
                label=f"{cell['execution_id']} marker k",
            )
            marker_error = _finite(
                marker.get("error"),
                label=f"{cell['execution_id']} marker error",
            )
            ax.scatter(
                [marker_k],
                [max(marker_error, 1.0e-16)],
                color=style["color"],
                marker=style["marker"],
                s=34,
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
            )
            tuple_overlay = _terminal_cost_plot_overlay(
                terminal=_mapping(
                    cell.get("terminal"),
                    label=f"{cell['execution_id']} terminal tuple",
                ),
                method=method,
            )
            if tuple_overlay is None:
                terminal_status = str(
                    _mapping(
                        cell.get("terminal"),
                        label=f"{cell['execution_id']} terminal status",
                    ).get("status", "")
                )
                if not (partial and terminal_status == "paused-local"):
                    raise ReportInputError(
                        f"{cell['execution_id']}: plotted curve lacks a "
                        "completed round-50 terminal tuple"
                    )
            else:
                tuple_overlays.append((method, tuple_overlay))
            observed_errors.extend(y)
        for raw_overlay in diagnostic_overlays:
            overlay = _mapping(raw_overlay, label="plot diagnostic overlay")
            if (
                overlay.get("representation") != representation
                or overlay.get("regime") != regime
            ):
                continue
            overlay_points = [
                _mapping(row, label="plot diagnostic overlay point")
                for row in _sequence(
                    overlay.get("points"),
                    label="plot diagnostic overlay points",
                )
            ]
            if not overlay_points:
                raise ReportInputError("plot diagnostic overlay has no points")
            overlay_x = [
                _integer(row.get("k"), label="plot diagnostic overlay round")
                for row in overlay_points
            ]
            overlay_y = [
                max(
                    _finite(
                        row.get("error"),
                        label="plot diagnostic overlay error",
                    ),
                    1.0e-16,
                )
                for row in overlay_points
            ]
            ax.plot(
                overlay_x,
                overlay_y,
                color=CUMULATIVE_PLATEAU_MACRO_STYLE["color"],
                linewidth=CUMULATIVE_PLATEAU_MACRO_STYLE["linewidth"],
                linestyle="-",
                zorder=4,
            )
            overlay_marker = _mapping(
                overlay.get("marker"),
                label="plot diagnostic overlay marker",
            )
            ax.scatter(
                [
                    _integer(
                        overlay_marker.get("k"),
                        label="plot diagnostic overlay marker round",
                    )
                ],
                [
                    max(
                        _finite(
                            overlay_marker.get("error"),
                            label="plot diagnostic overlay marker error",
                        ),
                        1.0e-16,
                    )
                ],
                color=CUMULATIVE_PLATEAU_MACRO_STYLE["color"],
                marker=CUMULATIVE_PLATEAU_MACRO_STYLE["marker"],
                s=38,
                edgecolor="white",
                linewidth=0.5,
                zorder=6,
            )
            observed_errors.extend(overlay_y)
        if pending:
            ax.text(
                0.5,
                0.50,
                "Awaiting validated\n50-round results",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#555555",
            )
            ax.set_ylim(1.0e-8, 1.0e1)
        elif partial and not observed_errors:
            ax.text(
                0.5,
                0.50,
                "Awaiting validated\n50-round results",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8.2,
                color="#555555",
            )
            ax.set_ylim(1.0e-8, 1.0e1)
        elif observed_errors:
            low = 10 ** math.floor(math.log10(min(observed_errors)))
            high = 10 ** math.ceil(math.log10(max(observed_errors)))
            if math.isclose(low, high):
                low /= 10
                high *= 10
            ax.set_ylim(low, high)
        annotation_y = 0.972
        for method, tuple_overlay in tuple_overlays:
            artist = ax.text(
                0.985,
                annotation_y,
                tuple_overlay,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=4.45,
                color=str(METHODS[method]["color"]),
                zorder=8,
            )
            artist.set_path_effects(
                [
                    path_effects.withStroke(
                        linewidth=1.35,
                        foreground="white",
                    ),
                    path_effects.Normal(),
                ]
            )
            annotation_y -= 0.078
        if partial and observed_errors and missing_methods:
            ax.text(
                0.985,
                annotation_y - 0.006,
                "Pending: "
                + ", ".join(
                    str(METHODS[method]["short"])
                    for method in missing_methods
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=5.25,
                color="#7A3333",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#D8B8B8",
                    "alpha": 0.82,
                    "pad": 1.2,
                },
            )
        ax.set_yscale("log")
        ax.set_xlim(0, 50)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
        )
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="major", color="#D8D8D8", linewidth=0.55)
        ax.grid(which="minor", axis="y", color="#EEEEEE", linewidth=0.35)
        ax.tick_params(axis="both", labelsize=7.1)
        ax.set_title(REGIME_LABELS[regime][1], fontsize=9.1, pad=3)
        if index >= 3:
            ax.set_xlabel("ADAPT iteration", fontsize=8.1)
        if index % 3 == 0:
            ax.set_ylabel(r"Same-cutoff $|\Delta E|$", fontsize=8.1)

    handles = [
        Line2D(
            [0],
            [0],
            color=METHODS[method]["color"],
            linewidth=METHODS[method]["linewidth"],
            marker=METHODS[method]["marker"],
            markersize=5,
            markeredgecolor="white",
            label=METHODS[method]["label"],
        )
        for method in METHOD_ORDER
    ]
    if any(
        _mapping(row, label="plot diagnostic overlay legend").get(
            "representation"
        )
        == representation
        for row in diagnostic_overlays
    ):
        handles.append(
            Line2D(
                [0],
                [0],
                color=CUMULATIVE_PLATEAU_MACRO_STYLE["color"],
                linewidth=CUMULATIVE_PLATEAU_MACRO_STYLE["linewidth"],
                marker=CUMULATIVE_PLATEAU_MACRO_STYLE["marker"],
                markersize=5,
                markeredgecolor="white",
                label=CUMULATIVE_PLATEAU_MACRO_STYLE["label"],
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        fontsize=7.8,
        title=(
            "Curve marker: first effective plateau when serialized; "
            "otherwise terminal point\n"
            "Colored tuple: terminal k=50 "
            "(N2q, D2q, Dc, W1q, S_alg)"
        ),
        title_fontsize=6.45,
    )
    if pending:
        fig.text(
            0.5,
            0.51,
            "PENDING - NOT PAPER EVIDENCE",
            ha="center",
            va="center",
            fontsize=31,
            color="#8B1A1A",
            alpha=0.12,
            rotation=28,
            weight="bold",
        )
    elif partial:
        fig.text(
            0.5,
            0.51,
            "PARTIAL PROGRESS - NOT PAPER EVIDENCE",
            ha="center",
            va="center",
            fontsize=25,
            color="#8B1A1A",
            alpha=0.10,
            rotation=28,
            weight="bold",
        )
    fig.savefig(vector_output, facecolor="white")
    fig.savefig(png_output, dpi=240, facecolor="white")
    plt.close(fig)


def _render_qiskit_plateau_append_comparison(
    *,
    diagnostic: Mapping[str, Any],
    always_diagnostic: Mapping[str, Any],
    append_cell: Mapping[str, Any],
    vector_output: Path,
    png_output: Path,
) -> None:
    """Render one single-axis convergence comparison for the diagnostic page."""

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

    diagnostic_points = [
        _mapping(row, label="Qiskit-plateau plotted point")
        for row in _sequence(
            diagnostic.get("points"),
            label="Qiskit-plateau plotted points",
        )
    ]
    append_points = [
        _mapping(row, label="Append plotted point")
        for row in _sequence(
            append_cell.get("points"),
            label="Append plotted points",
        )
    ]
    always_points = [
        _mapping(row, label="Qiskit-always plotted point")
        for row in _sequence(
            always_diagnostic.get("points"),
            label="Qiskit-always plotted points",
        )
    ]
    if (
        [int(row["k"]) for row in diagnostic_points] != list(range(51))
        or [int(row["k"]) for row in append_points] != list(range(51))
        or [int(row["k"]) for row in always_points] != list(range(14))
    ):
        raise ReportInputError(
            "Qiskit comparison curve horizons drifted"
        )
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.55, 4.45), dpi=240)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.83, bottom=0.15)
    styles = {
        "diagnostic": {
            "color": "#7B2CBF",
            "linewidth": 2.15,
            "marker": "D",
            "label": (
                "Stationary RA: plateau insertion, all-phase "
                "Qiskit cost ranking"
            ),
        },
        "append": {
            "color": "#4C78A8",
            "linewidth": 1.8,
            "marker": "o",
            "label": "Conventional unwhitened Append-ADAPT",
        },
        "always": {
            "color": "#E45756",
            "linewidth": 2.05,
            "marker": "*",
            "label": (
                "Stationary RA: always insertion, all-phase "
                "Qiskit cost ranking"
            ),
        },
    }
    for key, points, marker in (
        (
            "diagnostic",
            diagnostic_points,
            _mapping(
                diagnostic.get("marker"),
                label="Qiskit-plateau comparison marker",
            ),
        ),
        (
            "always",
            always_points,
            _mapping(
                always_diagnostic.get("marker"),
                label="Qiskit-always comparison marker",
            ),
        ),
        (
            "append",
            append_points,
            _mapping(
                append_cell.get("marker"),
                label="Append comparison marker",
            ),
        ),
    ):
        style = styles[key]
        x = [int(row["k"]) for row in points]
        y = [max(float(row["error"]), 1.0e-16) for row in points]
        ax.plot(
            x,
            y,
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle="-",
        )
        marker_k = _integer(
            marker.get("k"),
            label=f"{key} comparison marker round",
        )
        marker_error = _finite(
            marker.get("error"),
            label=f"{key} comparison marker error",
        )
        ax.scatter(
            [marker_k],
            [max(marker_error, 1.0e-16)],
            color=style["color"],
            marker=style["marker"],
            s=54,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
    diagnostic_terminal = _mapping(
        diagnostic.get("terminal"),
        label="Qiskit-plateau comparison terminal",
    )
    append_terminal = _mapping(
        append_cell.get("terminal"),
        label="Append comparison terminal",
    )
    always_terminal = _mapping(
        always_diagnostic.get("terminal"),
        label="Qiskit-always comparison terminal",
    )
    comparison = _mapping(
        diagnostic.get("comparison"),
        label="Qiskit-plateau comparison metrics",
    )
    ax.text(
        0.985,
        0.965,
        (
            "Observed endpoints\n"
            rf"plateau RA, k=50: $|\Delta E|="
            f"{float(diagnostic_terminal['error']):.3e}$\n"
            rf"always RA, k=13: $|\Delta E|="
            f"{float(always_terminal['error']):.3e}$\n"
            rf"Append, k=50: $|\Delta E|="
            f"{float(append_terminal['error']):.3e}$\n"
            rf"plateau/Append ratio: "
            f"{float(comparison['terminal_error_ratio_qiskit_ra_over_append']):.6f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color="#222222",
        bbox={
            "facecolor": "white",
            "edgecolor": "#B8B8B8",
            "alpha": 0.92,
            "pad": 3.2,
        },
    )
    fixed_rows = (
        (
            r"$\diamond$ plateau: ",
            _mapping(
                diagnostic.get("fixed_iteration_qiskit"),
                label="Qiskit-plateau fixed tuple overlay",
            ),
        ),
        (
            r"$\star$ always: ",
            _mapping(
                always_diagnostic.get("fixed_iteration_qiskit"),
                label="Qiskit-always fixed tuple overlay",
            ),
        ),
        (
            r"$\bullet$ Append: ",
            _mapping(
                append_cell.get("fixed_iteration_qiskit"),
                label="Append fixed tuple overlay",
            ),
        ),
    )
    fixed_text = "Iteration 10 compiled tuples\n" + "\n".join(
        label
        + paper_i_cost_tuple_latex(
            observation,
            marker="",
            format_s_alg=_format_s_alg_mathtext,
        )
        for label, observation in fixed_rows
    )
    ax.text(
        0.985,
        0.055,
        fixed_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.15,
        color="#222222",
        bbox={
            "facecolor": "white",
            "edgecolor": "#B8B8B8",
            "alpha": 0.92,
            "pad": 3.0,
        },
    )
    ax.set_yscale("log")
    ax.set_xlim(0, 50)
    all_errors = [
        max(float(row["error"]), 1.0e-16)
        for row in (*diagnostic_points, *always_points, *append_points)
    ]
    low = 10 ** math.floor(math.log10(min(all_errors)))
    high = 10 ** math.ceil(math.log10(max(all_errors)))
    ax.set_ylim(low, high)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=11, integer=True))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(
        LogLocator(
            base=10,
            subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        )
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", color="#D8D8D8", linewidth=0.6)
    ax.grid(which="minor", axis="y", color="#EEEEEE", linewidth=0.35)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.set_xlabel("ADAPT iteration", fontsize=10)
    ax.set_ylabel(r"Same-cutoff $|\Delta E|$", fontsize=10)
    handles = [
        Line2D(
            [0],
            [0],
            color=styles[key]["color"],
            linewidth=styles[key]["linewidth"],
            marker=styles[key]["marker"],
            markersize=6,
            markeredgecolor="white",
            label=styles[key]["label"],
        )
        for key in ("diagnostic", "always", "append")
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=1,
        frameon=False,
        fontsize=8.4,
        title=(
            "Strong--weak macro, nph=3; one marker per curve "
            "(RA: effective prefix, ADAPT: serialized terminal)"
        ),
        title_fontsize=7.6,
    )
    fig.text(
        0.5,
        0.50,
        "DIAGNOSTIC - NOT PAPER EVIDENCE",
        ha="center",
        va="center",
        fontsize=27,
        color="#8B1A1A",
        alpha=0.085,
        rotation=27,
        weight="bold",
    )
    fig.savefig(vector_output, facecolor="white")
    fig.savefig(png_output, dpi=240, facecolor="white")
    plt.close(fig)


def _render_global_singleton_weak_weak_comparison(
    *,
    diagnostic: Mapping[str, Any],
    vector_output: Path,
    png_output: Path,
) -> None:
    """Render the separate matched weak--weak insertion-policy diagnostic."""

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

    arms = _mapping(
        diagnostic.get("arms_by_policy"),
        label="global-singleton weak-weak arms",
    )
    styles = {
        "append_commutation_reduced": {
            "color": "#4C78A8",
            "linewidth": 1.9,
            "marker": "o",
            "label": "RA global singleton: append placement",
        },
        "plateau_commutation": {
            "color": "#7B2CBF",
            "linewidth": 2.1,
            "marker": "D",
            "label": "RA global singleton: plateau placement",
        },
    }
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.55, 4.15), dpi=240)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.16)
    all_errors: list[float] = []
    for policy in GLOBAL_SINGLETON_WW_POLICIES:
        arm = _mapping(
            arms.get(policy),
            label=f"global-singleton {policy} arm",
        )
        points = [
            _mapping(row, label=f"global-singleton {policy} point")
            for row in _sequence(
                arm.get("points"),
                label=f"global-singleton {policy} points",
            )
        ]
        style = styles[policy]
        x = [int(row["k"]) for row in points]
        y = [max(float(row["error"]), 1.0e-16) for row in points]
        all_errors.extend(y)
        ax.plot(
            x,
            y,
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle="-",
        )
        effective = _mapping(
            arm.get("effective_plateau"),
            label=f"global-singleton {policy} effective plateau",
        )
        ax.scatter(
            [int(effective["k"])],
            [max(float(effective["error"]), 1.0e-16)],
            color=style["color"],
            marker=style["marker"],
            s=58,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
    append = _mapping(
        arms.get("append_commutation_reduced"),
        label="global-singleton append arm",
    )
    plateau = _mapping(
        arms.get("plateau_commutation"),
        label="global-singleton plateau arm",
    )
    append_terminal = _mapping(
        append.get("terminal"),
        label="global-singleton append terminal",
    )
    plateau_terminal = _mapping(
        plateau.get("terminal"),
        label="global-singleton plateau terminal",
    )
    derived = _mapping(
        diagnostic.get("derived"),
        label="global-singleton weak-weak derived comparison",
    )
    plateau_insertions = _mapping(
        plateau.get("insertion_counts"),
        label="global-singleton plateau insertion counts",
    )
    ax.text(
        0.985,
        0.965,
        (
            "Observed k=50 endpoints\n"
            rf"append: $|\Delta E|={float(append_terminal['error']):.3e}$, "
            rf"$S_{{\rm alg}}={int(append_terminal['S_alg']):,}$"
            "\n"
            rf"plateau: $|\Delta E|={float(plateau_terminal['error']):.3e}$, "
            rf"$S_{{\rm alg}}={int(plateau_terminal['S_alg']):,}$"
            "\n"
            rf"plateau/append work: "
            f"{float(derived['terminal_s_alg_ratio_plateau_over_append']):.3f}"
            r"$\times$"
            "\n"
            f"plateau interior placements: "
            f"{int(plateau_insertions['interior_count'])}/50"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.15,
        color="#222222",
        bbox={
            "facecolor": "white",
            "edgecolor": "#B8B8B8",
            "alpha": 0.92,
            "pad": 3.2,
        },
    )
    ax.set_yscale("log")
    ax.set_xlim(0, 50)
    low = 10 ** math.floor(math.log10(min(all_errors)))
    high = 10 ** math.ceil(math.log10(max(all_errors)))
    ax.set_ylim(low, high)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=11, integer=True))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(
        LogLocator(
            base=10,
            subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        )
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", color="#D8D8D8", linewidth=0.6)
    ax.grid(which="minor", axis="y", color="#EEEEEE", linewidth=0.35)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.set_xlabel("RA-ADAPT controller round", fontsize=10)
    ax.set_ylabel(r"Same-cutoff $|\Delta E|$", fontsize=10)
    handles = [
        Line2D(
            [0],
            [0],
            color=styles[policy]["color"],
            linewidth=styles[policy]["linewidth"],
            marker=styles[policy]["marker"],
            markersize=6,
            markeredgecolor="white",
            label=styles[policy]["label"],
        )
        for policy in GLOBAL_SINGLETON_WW_POLICIES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        title=(
            "Weak--weak, nph=3; stationary gradients + "
            "all-phase resource weighting"
        ),
        title_fontsize=7.7,
    )
    fig.text(
        0.5,
        0.50,
        "SEPARATE DIAGNOSTIC - NOT PAPER EVIDENCE",
        ha="center",
        va="center",
        fontsize=24,
        color="#8B1A1A",
        alpha=0.085,
        rotation=27,
        weight="bold",
    )
    fig.savefig(vector_output, facecolor="white")
    fig.savefig(png_output, dpi=240, facecolor="white")
    plt.close(fig)


def _render_qiskit_singleton_round33_comparison(
    *,
    diagnostic: Mapping[str, Any],
    vector_output: Path,
    png_output: Path,
) -> None:
    """Render the matched-round strong--strong singleton comparison."""

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

    baselines = _mapping(
        diagnostic.get("baseline_curves"),
        label="singleton round-33 baseline curves",
    )
    observations = _mapping(
        diagnostic.get("matched_observations"),
        label="singleton round-33 observations",
    )
    curves = {
        "global_singleton_qiskit_ra": _sequence(
            diagnostic.get("points"),
            label="singleton round-33 Qiskit-ranked points",
        ),
        "staged_proxy_ra_plateau": _sequence(
            baselines.get("staged_proxy_ra_plateau"),
            label="singleton round-33 staged RA points",
        ),
        "conventional_append_adapt": _sequence(
            baselines.get("conventional_append_adapt"),
            label="singleton round-33 Append points",
        ),
    }
    if any(
        [int(_mapping(row, label="singleton plotted point")["k"])
         for row in points]
        != list(range(MATCHED_SINGLETON_ROUND + 1))
        for points in curves.values()
    ):
        raise ReportInputError(
            "singleton round-33 comparison curve horizons drifted"
        )
    styles = {
        "global_singleton_qiskit_ra": {
            "color": "#7B2CBF",
            "linewidth": 2.2,
            "marker": "D",
            "label": "Global-singleton RA: all-phase Qiskit ranking",
        },
        "staged_proxy_ra_plateau": {
            "color": "#E45756",
            "linewidth": 1.9,
            "marker": "s",
            "label": "Staged singleton RA-plateau: Paper-I proxy ranking",
        },
        "conventional_append_adapt": {
            "color": "#4C78A8",
            "linewidth": 1.8,
            "marker": "o",
            "label": "Conventional unwhitened Append-ADAPT",
        },
    }
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.55, 4.25), dpi=240)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.81, bottom=0.16)
    all_errors: list[float] = []
    for key, raw_points in curves.items():
        points = [
            _mapping(row, label=f"singleton round-33 {key} point")
            for row in raw_points
        ]
        style = styles[key]
        x = [int(row["k"]) for row in points]
        y = [max(float(row["error"]), 1.0e-16) for row in points]
        all_errors.extend(y)
        ax.plot(
            x,
            y,
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle="-",
        )
        observation = _mapping(
            observations.get(key),
            label=f"singleton round-33 {key} observation",
        )
        ax.scatter(
            [MATCHED_SINGLETON_ROUND],
            [max(float(observation["error"]), 1.0e-16)],
            color=style["color"],
            marker=style["marker"],
            s=58,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
    error_text = "Matched k=33 errors\n" + "\n".join(
        (
            f"{label}: $|\\Delta E|={float(observations[key]['error']):.3e}$"
        )
        for key, label in (
            ("global_singleton_qiskit_ra", "global Qiskit RA"),
            ("staged_proxy_ra_plateau", "staged proxy RA"),
            ("conventional_append_adapt", "Append-ADAPT"),
        )
    )
    ax.text(
        0.985,
        0.965,
        error_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.9,
        color="#222222",
        bbox={
            "facecolor": "white",
            "edgecolor": "#B8B8B8",
            "alpha": 0.92,
            "pad": 3.1,
        },
    )
    tuple_text = "Matched k=33 compiled tuples\n" + "\n".join(
        label
        + paper_i_cost_tuple_latex(
            _mapping(
                observations.get(key),
                label=f"singleton round-33 {key} tuple",
            ),
            marker="",
            format_s_alg=_format_s_alg_mathtext,
        )
        for key, label in (
            ("global_singleton_qiskit_ra", r"$\diamond$ global: "),
            ("staged_proxy_ra_plateau", r"$\boxminus$ staged: "),
            ("conventional_append_adapt", r"$\bullet$ Append: "),
        )
    )
    ax.text(
        0.985,
        0.055,
        tuple_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#222222",
        bbox={
            "facecolor": "white",
            "edgecolor": "#B8B8B8",
            "alpha": 0.92,
            "pad": 3.0,
        },
    )
    ax.set_yscale("log")
    ax.set_xlim(-0.25, MATCHED_SINGLETON_ROUND + 0.75)
    low = 10 ** math.floor(math.log10(min(all_errors)))
    high = 10 ** math.ceil(math.log10(max(all_errors)))
    ax.set_ylim(low, high)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(
        LogLocator(
            base=10,
            subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        )
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", color="#D8D8D8", linewidth=0.6)
    ax.grid(which="minor", axis="y", color="#EEEEEE", linewidth=0.35)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.set_xlabel("ADAPT iteration", fontsize=10)
    ax.set_ylabel(r"Same-cutoff $|\Delta E|$", fontsize=10)
    handles = [
        Line2D(
            [0],
            [0],
            color=styles[key]["color"],
            linewidth=styles[key]["linewidth"],
            marker=styles[key]["marker"],
            markersize=6,
            markeredgecolor="white",
            label=styles[key]["label"],
        )
        for key in (
            "global_singleton_qiskit_ra",
            "staged_proxy_ra_plateau",
            "conventional_append_adapt",
        )
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=1,
        frameon=False,
        fontsize=8.1,
        title=(
            "Strong--strong singleton, nph=7; one matched-round marker "
            "per curve at k=33"
        ),
        title_fontsize=7.5,
    )
    fig.text(
        0.5,
        0.50,
        "DIAGNOSTIC - NOT PAPER EVIDENCE",
        ha="center",
        va="center",
        fontsize=25,
        color="#8B1A1A",
        alpha=0.08,
        rotation=27,
        weight="bold",
    )
    fig.savefig(vector_output, facecolor="white")
    fig.savefig(png_output, dpi=240, facecolor="white")
    plt.close(fig)


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


def _format_error_tex(value: Any) -> str:
    if value is None:
        return "--"
    numeric = float(value)
    if numeric == 0.0:
        return "$0$"
    mantissa, exponent = f"{numeric:.2e}".split("e")
    return rf"${mantissa}\mathord{{\times}}10^{{{int(exponent)}}}$"


def _format_s_alg_tex(value: int) -> str:
    """Format estimator work as compact two-significant-digit e notation."""

    mantissa, exponent = f"{int(value):.1e}".split("e")
    return rf"{mantissa}\mathrm{{e}}{int(exponent)}"


def _tex_path(value: Any) -> str:
    text = str(value)
    if "{" in text or "}" in text or "\n" in text:
        raise ReportInputError("LaTeX path value contains unsafe delimiters")
    return rf"\url{{{text}}}"


def _terminal_cost_tuple_tex(terminal: Mapping[str, Any]) -> str:
    if not str(terminal.get("status", "")).startswith("complete"):
        return "--"
    try:
        return paper_i_cost_tuple_latex(
            terminal,
            marker="",
            format_s_alg=_format_s_alg_tex,
        )
    except (TypeError, ValueError) as exc:
        raise ReportInputError(
            f"terminal Paper-I Qiskit cost tuple is incomplete: {exc}"
        ) from exc


def _terminal_table_tex(
    *,
    cells: Sequence[Mapping[str, Any]],
    representation_key: str,
) -> str:
    representation = REPRESENTATIONS[representation_key]
    indexed = _cell_index(cells)
    rows = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrl}",
        r"\toprule",
        (
            r"Reg. & Route & $k_T$ & $|\Delta E_T|$ & "
            rf"${PAPER_I_QISKIT_COST_TUPLE_LATEX}$ & status \\"
        ),
        r"\midrule",
    ]
    for regime_index, regime in enumerate(REGIME_ORDER):
        for method in METHOD_ORDER:
            terminal = indexed[(representation, regime, method)]["terminal"]
            rows.append(
                " & ".join(
                    (
                        _tex_escape(REGIME_LABELS[regime][0]),
                        _tex_escape(METHODS[method]["short"]),
                        _tex_escape(_format_integer(terminal.get("k"))),
                        _format_error_tex(terminal.get("error")),
                        _terminal_cost_tuple_tex(terminal),
                        _tex_escape(terminal.get("status", "pending")),
                    )
                )
                + r" \\"
            )
        if regime_index != len(REGIME_ORDER) - 1:
            rows.append(r"\addlinespace[0.18ex]")
    rows.extend((r"\bottomrule", r"\end{tabular*}"))
    return "\n".join(rows)


def _parameter_manifest_tex(manifest: Mapping[str, Any]) -> str:
    regimes = _sequence(manifest.get("regimes"), label="manifest regimes")
    regime_text = "; ".join(
        (
            f"{REGIME_LABELS[str(row['regime_id'])][0]}: "
            f"U={float(row['u']):.6g}, "
            f"omega0={float(row['omega0']):.6g}, "
            f"g={float(row['g_ep']):.6g}, "
            f"nph={int(row['n_ph_max'])}"
        )
        for row in (
            _mapping(raw, label="manifest regime") for raw in regimes
        )
    )
    package_sources = _mapping(
        manifest.get("package_provenance"),
        label="manifest package provenance",
    )
    if package_sources.get("cross_revision") is True:
        source_lines = []
        first_source_by_package_binding: dict[
            tuple[str, str, str], int
        ] = {}
        for raw in _sequence(
            package_sources.get("sources"),
            label="cross-revision manifest sources",
        ):
            source = _mapping(
                raw, label="cross-revision manifest source"
            )
            source_index = int(source["source_receipt_index"])
            package_binding = (
                str(source["package_id"]),
                str(source["core_materialization_id"]),
                str(source["source_archive_sha256"]),
            )
            first_source_index = first_source_by_package_binding.get(
                package_binding
            )
            if first_source_index is None:
                first_source_by_package_binding[package_binding] = (
                    source_index
                )
                source_lines.append(
                    r"\par "
                    + _tex_escape(
                        f"source[{source_index}] "
                        f"{source['method_family']}: "
                    )
                    + _tex_path(source["package_id"])
                    + "; core="
                    + _tex_path(source["core_materialization_id"])
                    + "; validation="
                    + _tex_path(source["validation_sha256"])
                    + "; archive="
                    + _tex_path(source["source_archive_sha256"])
                    + "."
                )
            else:
                source_lines.append(
                    r"\par "
                    + _tex_escape(
                        f"source[{source_index}] "
                        f"{source['method_family']}: "
                        "package/core/archive as "
                        f"source[{first_source_index}]; "
                    )
                    + "validation="
                    + _tex_path(source["validation_sha256"])
                    + "."
                )
        for raw in _sequence(
            package_sources.get("recovery_adapters", ()),
            label="cross-revision recovery adapters",
        ):
            recovery = _mapping(
                raw,
                label="cross-revision recovery adapter",
            )
            counts = _mapping(
                recovery.get("recovery_counts"),
                label="cross-revision recovery counts",
            )
            source_lines.append(
                r"\par "
                + _tex_escape(
                    f"source[{recovery['source_receipt_index']}] "
                    "explicit recovery adapter: "
                )
                + _tex_path(recovery["sha256"])
                + _tex_escape(
                    "; "
                    f"{recovery['included_count']} observations "
                    f"({counts.get(RECOVERY_CROSS_CAMPAIGN_CLASS, 0)} "
                    "passed cross-campaign equivalents; "
                    f"{counts.get(RECOVERY_G5_UNEXERCISED_CLASS, 0)} "
                    "completed G5-unexercised trajectories); "
                    "not paper evidence."
                )
            )
        for raw in _sequence(
            package_sources.get("local_paused_prefixes", ()),
            label="cross-revision local paused prefixes",
        ):
            paused = _mapping(
                raw,
                label="cross-revision local paused prefix",
            )
            checkpoint = _mapping(
                paused.get("checkpoint"),
                label="cross-revision paused checkpoint",
            )
            source_lines.append(
                r"\par "
                + _tex_escape(
                    "local paused prefix "
                    f"{paused['execution_id']}: k="
                    f"{paused['paused_controller_round']}; "
                )
                + "checkpoint="
                + _tex_path(checkpoint["sha256"])
                + _tex_escape("; diagnostic only, no round-50 cost tuple.")
            )
        authority_tex = (
            r"\par \textbf{Cross-revision source receipts "
            r"(exact digests):}"
            + "".join(source_lines)
        )
    else:
        authority_tex = (
            r"\par \textbf{Authority:} "
            + _tex_path(CORE_MATERIALIZATION_ID)
            + r"; package="
            + _tex_path(PACKAGE_ID)
            + r"; manifest="
            + _tex_path(package_sources["package_manifest"]["sha256"])
            + "."
        )
    return (
        r"\fcolorbox{black!35}{black!2}{%"
        "\n"
        r"\begin{minipage}{0.975\textwidth}"
        "\n"
        r"\raggedright"
        "\n"
        r"\fontsize{6.25}{7.15}\selectfont"
        "\n"
        r"\textbf{Parameter and provenance manifest.} "
        + _tex_escape(
            f"Model={manifest['model_family']}; L={manifest['num_sites']}; "
            f"boundary={manifest['boundary']}; "
            f"sector={manifest['sector_label']}; "
            f"bosons={manifest['boson_encoding']}; "
            f"drive_enabled={str(manifest['drive_enabled']).lower()}; "
            f"zero_point={str(manifest['include_zero_point']).lower()}."
        )
        + r"\par "
        + _tex_escape(
            f"Horizon={manifest['horizon']}; optimizer="
            f"{manifest['optimizer']}-{manifest['optimizer_maxiter']}; "
            f"seeds={manifest['seeds']}; active gradients="
            f"{manifest['active_gradient_policy']}; weighting="
            f"{manifest['resource_weighting_scope']}; exact target="
            f"{manifest['exact_target_label']} at identical n_ph_max."
        )
        + r"\par "
        + _tex_escape(regime_text)
        + authority_tex
        + "\n"
        r"\end{minipage}}"
    )


def _report_page_tex(
    *,
    cells: Sequence[Mapping[str, Any]],
    representation_key: str,
    plot_pdf: Path,
    manifest: Mapping[str, Any],
    include_manifest: bool,
    pending: bool,
    partial: bool = False,
) -> str:
    if pending and partial:
        raise ReportInputError("report page cannot be both pending and partial")
    plot_height = "4.32in"
    manifest_block = (
        _parameter_manifest_tex(manifest)
        if include_manifest
        else (
            (
                r"{\fontsize{6.25}{7.15}\selectfont "
                r"Cross-revision sources: exact package, validation, and "
                r"archive digests on page 1 and in the provenance sidecar; "
                r"same-cutoff ED reference.}"
            )
            if _mapping(
                manifest.get("package_provenance"),
                label="manifest package provenance",
            ).get("cross_revision")
            is True
            else (
                r"{\fontsize{6.25}{7.15}\selectfont "
                + r"Authority: "
                + _tex_path(CORE_MATERIALIZATION_ID)
                + r"; package: "
                + _tex_path(PACKAGE_ID)
                + r"; same-cutoff ED reference."
                + r"}"
            )
        )
    )
    included_count = sum(bool(row.get("points")) for row in cells)
    cost_tuple_heading = (
        r"Round-50 same-cutoff error and cost vector $"
        + PAPER_I_QISKIT_COST_TUPLE_LATEX
        + r"$"
    )
    cost_tuple_note = (
        r"Colored in-panel tuples and table rows are evaluated at terminal "
        r"$k=50$; curve markers do not select their costs. "
        r"$W_{1q}$ is Qiskit-emitted pretranspilation Pauli one-qubit "
        r"work ($H+S+S^\dagger+R_z$), excluding reference preparation; "
        r"$S_{\rm alg}$ is closed logical estimator work."
    )
    if pending:
        status_line = (
            r"\textcolor{red!65!black}{\textbf{"
            r"PENDING - NOT PAPER EVIDENCE}}"
        )
        terminal_heading = cost_tuple_heading
        footer = (
            r"Marker: first effective plateau prefix when serialized; "
            r"otherwise terminal plotted point. "
            + cost_tuple_note
        )
    elif partial:
        package_provenance = _mapping(
            manifest.get("package_provenance"),
            label="manifest package provenance",
        )
        cross_revision = package_provenance.get("cross_revision") is True
        recovery_count = _integer(
            package_provenance.get("recovery_cell_count", 0),
            label="manifest recovery cell count",
        )
        local_paused_count = _integer(
            package_provenance.get("local_paused_prefix_count", 0),
            label="manifest local paused-prefix count",
        )
        if recovery_count or local_paused_count:
            plot_height = "4.08in"
        validated_count = (
            included_count - recovery_count - local_paused_count
        )
        if validated_count < 0:
            raise ReportInputError(
                "manifest recovery count exceeds included count"
            )
        status_line = (
            r"\textcolor{red!65!black}{\textbf{"
            + _tex_escape(
                (
                    "PARTIAL CROSS-REVISION PROGRESS"
                    if cross_revision
                    else "PARTIAL PROGRESS"
                )
                + " - NOT PAPER EVIDENCE "
                + (
                    f"({included_count}/48 observed: "
                    f"{validated_count} validated + "
                    f"{recovery_count} recovered; "
                    if recovery_count
                    else f"({included_count}/48 validated; "
                )
                + (
                    f"{local_paused_count} paused local prefixes; "
                    if local_paused_count
                    else ""
                )
                + f"{48 - included_count} pending)"
            )
            + "}}"
        )
        terminal_heading = r"Available " + cost_tuple_heading
        footer = (
            (
                r"Append and RA curves retain disjoint, explicit source "
                r"package identities. "
                if cross_revision
                else ""
            )
            + (
                r"Most included curves are passed validated attempts. "
                r"Rows marked complete-Xrev are passed cross-campaign "
                r"science-equivalent RA-always observations; rows marked "
                r"complete-G5* completed 50 rounds and passed every other "
                r"scientific gate, but the plateau interior domain remained "
                r"unexercised. Both recovery classes are diagnostic only. "
                if recovery_count
                else
                r"Included curves are passed validated attempts with full "
                r"source-lock and same-cutoff closure. "
            )
            + (
                r"Rows marked paused-local are authenticated accepted "
                r"prefixes from deliberately stopped local runs; they have "
                r"no round-50 compiled tuple and are diagnostic only. "
                if local_paused_count
                else ""
            )
            + r"Dashes are pending cells; "
            r"this partial report cannot adopt paper evidence. "
            r"Curve markers show effective plateaus when serialized and "
            r"otherwise terminal points. "
            + cost_tuple_note
        )
    else:
        status_line = (
            r"Validated selected attempts; display does not itself adopt "
            r"paper evidence."
        )
        terminal_heading = cost_tuple_heading
        footer = (
            r"Marker: first effective plateau prefix when serialized; "
            r"otherwise terminal plotted point. "
            + cost_tuple_note
        )
    return rf"""
\begin{{center}}
{{\large\bfseries {_tex_escape(REPRESENTATION_TITLES[representation_key])}}}\\[-0.2ex]
{{\fontsize{{7.2}}{{8.2}}\selectfont Six Hubbard-Holstein regimes; same-cutoff absolute energy error. {status_line}}}
\end{{center}}
\vspace{{0.25ex}}
{manifest_block}
\vspace{{0.35ex}}
\begin{{center}}
\includegraphics[width=0.995\textwidth,height={plot_height},keepaspectratio]{{{_tex_escape(plot_pdf.name)}}}
\end{{center}}
\vspace{{-1.2ex}}
{{\fontsize{{7.1}}{{8.0}}\selectfont\bfseries {terminal_heading}\par}}
\vspace{{0.25ex}}
{{\fontsize{{5.35}}{{5.85}}\selectfont
{_terminal_table_tex(cells=cells, representation_key=representation_key)}
}}
\vfill
{{\fontsize{{5.5}}{{6.2}}\selectfont {footer}}}
"""


def _qiskit_plateau_comparison_page_tex(
    *,
    diagnostic: Mapping[str, Any],
    always_diagnostic: Mapping[str, Any],
    append_cell: Mapping[str, Any],
    proxy_comparison: Mapping[str, Any],
    plot_pdf: Path,
) -> str:
    terminal = _mapping(
        diagnostic.get("terminal"),
        label="Qiskit-plateau diagnostic terminal",
    )
    insertion = _mapping(
        diagnostic.get("insertion"),
        label="Qiskit-plateau diagnostic insertion",
    )
    selector = _mapping(
        diagnostic.get("online_qiskit_selector"),
        label="Qiskit-plateau online selector",
    )
    always_terminal = _mapping(
        always_diagnostic.get("terminal"),
        label="Qiskit-always diagnostic terminal",
    )
    append_terminal = _mapping(
        append_cell.get("terminal"),
        label="Qiskit-plateau matched Append terminal",
    )
    exact_energy = _finite(
        diagnostic.get("same_cutoff_exact_energy"),
        label="Qiskit-plateau same-cutoff ED energy",
    )
    fixed = {
        "qiskit_plateau": _mapping(
            diagnostic.get("fixed_iteration_qiskit"),
            label="Qiskit-ranked plateau fixed iteration",
        ),
        "qiskit_always": _mapping(
            always_diagnostic.get("fixed_iteration_qiskit"),
            label="Qiskit-ranked always fixed iteration",
        ),
        "append": _mapping(
            append_cell.get("fixed_iteration_qiskit"),
            label="Append fixed iteration",
        ),
        "proxy_plateau": _mapping(
            proxy_comparison.get("plateau"),
            label="proxy-ranked plateau fixed iteration",
        ),
        "proxy_none": _mapping(
            proxy_comparison.get("no_insertion"),
            label="proxy-ranked no-insertion fixed iteration",
        ),
    }
    plateau_depth_reduction = (
        1.0
        - _integer(
            fixed["qiskit_plateau"].get("D2q"),
            label="Qiskit-ranked plateau D2q",
        )
        / _integer(
            fixed["proxy_plateau"].get("D2q"),
            label="proxy-ranked plateau D2q",
            minimum=1,
        )
    ) * 100.0
    plateau_total_depth_reduction = (
        1.0
        - _integer(
            fixed["qiskit_plateau"].get("Dc"),
            label="Qiskit-ranked plateau Dc",
        )
        / _integer(
            fixed["proxy_plateau"].get("Dc"),
            label="proxy-ranked plateau Dc",
            minimum=1,
        )
    ) * 100.0

    def row(label: str, observation: Mapping[str, Any]) -> str:
        return (
            f"{label} & "
            f"{_format_error_tex(observation['error'])} & "
            f"${_format_s_alg_tex(int(observation['S_alg']))}$ & "
            f"{_terminal_cost_tuple_tex(observation)} \\\\"
        )

    return rf"""
\begin{{center}}
{{\large\bfseries Strong--weak macro diagnostic: full-transpile ranking versus proxy ranking}}\\[-0.1ex]
{{\fontsize{{7.25}}{{8.2}}\selectfont Plateau RA completed 50/50 rounds; commutation-reduced always RA completed its requested 13/13 rounds. Common comparison is iteration 10.}}
\end{{center}}
\vspace{{0.25ex}}
\fcolorbox{{black!35}}{{black!2}}{{%
\begin{{minipage}}{{0.975\textwidth}}
\fontsize{{6.35}}{{7.2}}\selectfont
\textbf{{Matched problem.}} Hubbard--Holstein $L=2$, strong--weak $U=8$,
$\omega_0=1$, $g=0.353553390593$, $n_{{\rm ph}}=3$, 8 qubits;
same-cutoff $E_{{\rm ED}}={exact_energy:.15f}$; macro-generator candidates;
Powell-200; seed 7.
\par
\textbf{{All-phase ranking.}} Stationary-source RA with full-trial Qiskit ranking
({_tex_escape(selector["backend"])}, optimization level
{int(selector["optimization_level"])}, transpiler seed
{int(selector["transpile_seed"])}). Plateau RA selected an interior position in
{int(insertion["interior_count"])}/50 rounds (first at round
{int(insertion["first_interior_round"])}); the separate always curve uses
commutation-reduced insertion at every round.
\end{{minipage}}}}
\vspace{{0.2ex}}
\begin{{center}}
\includegraphics[width=0.985\textwidth,height=3.75in,keepaspectratio]{{{_tex_escape(plot_pdf.name)}}}
\end{{center}}
\vspace{{-0.55ex}}
{{\fontsize{{6.65}}{{7.45}}\selectfont
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lrrl}}
\toprule
Route/ranking at $k=10$ & same-cutoff $|\Delta E_{{10}}|$ &
$S_{{\mathrm{{alg}},10}}$ &
$(N_{{2q}},D_{{2q}},D_c,W_{{1q}},S_{{\rm alg}})$ \\
\midrule
{row("All-phase Qiskit: plateau RA", fixed["qiskit_plateau"])}
{row("All-phase Qiskit: always RA", fixed["qiskit_always"])}
{row("Conventional Append-ADAPT", fixed["append"])}
{row("Page-1 proxy: plateau RA", fixed["proxy_plateau"])}
{row("Page-1 proxy: no-insertion RA", fixed["proxy_none"])}
Page-1 proxy: always RA & -- & -- & unavailable (run pending) \\
\bottomrule
\end{{tabular*}}
}}
\vspace{{0.35ex}}
{{\fontsize{{6.25}}{{7.15}}\selectfont
\textbf{{Like-for-like plateau result.}} At the same iteration and essentially
the same error, all-phase Qiskit ranking preserves $N_{{2q}}=200$ while
reducing $D_{{2q}}$ by {plateau_depth_reduction:.1f}\% and $D_c$ by
{plateau_total_depth_reduction:.1f}\% relative to page-1 proxy ranking.
\par
\textbf{{Interpretation boundary.}} This is a whole-route diagnostic, not an
insertion-only ablation: the RA route also uses stationary Phase-III
gradients, supported-metric trust, whitening, and a different cost-ranking
scope. The proxy-ranked always cell is genuinely unavailable. Plateau ends at
$k=50$ with $|\Delta E|={float(terminal["error"]):.3e}$; always ends at
$k=13$ with $|\Delta E|={float(always_terminal["error"]):.3e}$; Append ends
at $k=50$ with $|\Delta E|={float(append_terminal["error"]):.3e}$.
Both local Qiskit-ranked runs remain outside the validated 48-cell evidence matrix.
}}
"""


def _qiskit_singleton_round33_page_tex(
    *,
    diagnostic: Mapping[str, Any],
    plot_pdf: Path,
) -> str:
    """Build the matched-round strong--strong singleton diagnostic page."""

    observations = _mapping(
        diagnostic.get("matched_observations"),
        label="singleton round-33 page observations",
    )
    current = _mapping(
        observations.get("global_singleton_qiskit_ra"),
        label="singleton round-33 page current observation",
    )
    staged = _mapping(
        observations.get("staged_proxy_ra_plateau"),
        label="singleton round-33 page staged observation",
    )
    append = _mapping(
        observations.get("conventional_append_adapt"),
        label="singleton round-33 page Append observation",
    )
    insertion = _mapping(
        diagnostic.get("insertion"),
        label="singleton round-33 page insertion",
    )
    selector = _mapping(
        diagnostic.get("online_qiskit_selector"),
        label="singleton round-33 page selector",
    )
    exact_energy = _finite(
        diagnostic.get("same_cutoff_exact_energy"),
        label="singleton round-33 page exact energy",
    )

    def row(label: str, observation: Mapping[str, Any]) -> str:
        return (
            f"{label} & "
            f"{_format_error_tex(observation['error'])} & "
            f"${_format_s_alg_tex(int(observation['S_alg']))}$ & "
            f"{_terminal_cost_tuple_tex(observation)} \\\\"
        )

    return rf"""
\begin{{center}}
{{\large\bfseries Strong--strong singleton: matched round-33 route comparison}}\\[-0.1ex]
{{\fontsize{{7.25}}{{8.2}}\selectfont Global-singleton all-phase Qiskit ranking versus staged RA-plateau and conventional Append-ADAPT.}}
\end{{center}}
\vspace{{0.25ex}}
\fcolorbox{{black!35}}{{black!2}}{{%
\begin{{minipage}}{{0.975\textwidth}}
\fontsize{{6.35}}{{7.2}}\selectfont
\textbf{{Matched problem.}} Hubbard--Holstein $L=2$, strong--strong $U=8$,
$\omega_0=1$, $g=0.790569415042$, $n_{{\rm ph}}=7$, 10 qubits;
same-cutoff $E_{{\rm ED}}={exact_energy:.15f}$; Powell-200; seed 7.
All three observations are evaluated after exactly 33 accepted generators.
\par
\textbf{{Common compiled-cost readout.}} Every round-33 prefix is rebuilt and
compiled through the same Paper-I
\texttt{{table\_i\_basis\_gate\_transpile\_v1}} path
({_tex_escape(selector['backend'])}, optimization level
{int(selector['optimization_level'])}, transpiler seed
{int(selector['transpile_seed'])}). The plotted marker is the common round,
not an independently selected plateau.
\end{{minipage}}}}
\vspace{{0.15ex}}
\begin{{center}}
\includegraphics[width=0.985\textwidth,height=3.65in,keepaspectratio]{{{_tex_escape(plot_pdf.name)}}}
\end{{center}}
\vspace{{-0.55ex}}
{{\fontsize{{6.55}}{{7.35}}\selectfont
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lrrl}}
\toprule
Route at $k=33$ & same-cutoff $|\Delta E_{{33}}|$ &
$S_{{\mathrm{{alg}},33}}$ &
$(N_{{2q}},D_{{2q}},D_c,W_{{1q}},S_{{\rm alg}})$ \\
\midrule
{row("Global singleton RA: all-phase Qiskit", current)}
{row("Staged singleton RA-plateau: proxy-ranked", staged)}
{row("Conventional unwhitened Append-ADAPT", append)}
\bottomrule
\end{{tabular*}}
}}
\vspace{{0.35ex}}
{{\fontsize{{6.2}}{{7.1}}\selectfont
\textbf{{Observed round-33 comparison.}} The global Qiskit-ranked route has
$|\Delta E|={float(current['error']):.3e}$ after 33 accepted rounds. Its
preserved checkpoint contains {int(insertion['interior_count'])} interior and
{int(insertion['append_position_count'])} append-position selections through
that round.
\par
\textbf{{Interpretation boundary.}} This is deliberately a matched-round,
whole-route comparison, not a one-variable cost-oracle ablation. The current
route exposes the full guarded singleton pool in Phase I and applies compiled
cost in all three phases; the staged RA baseline shortlists macro parents
before singleton children and applies the Paper-I proxy/late-cost policy.
Append-ADAPT is conventional and unwhitened. The current run is an authenticated
33-round checkpoint from an interrupted 50-round local objective, remains
outside the validated 48-cell matrix, and is not claimed as terminal evidence.
}}
"""


def _global_singleton_weak_weak_page_tex(
    *,
    diagnostic: Mapping[str, Any],
    plot_pdf: Path,
) -> str:
    """Build the explicitly separated weak--weak insertion diagnostic page."""

    arms = _mapping(
        diagnostic.get("arms_by_policy"),
        label="global-singleton weak-weak arms",
    )
    append = _mapping(
        arms.get("append_commutation_reduced"),
        label="global-singleton append arm",
    )
    plateau = _mapping(
        arms.get("plateau_commutation"),
        label="global-singleton plateau arm",
    )
    append_terminal = _mapping(
        append.get("terminal"),
        label="global-singleton append terminal",
    )
    plateau_terminal = _mapping(
        plateau.get("terminal"),
        label="global-singleton plateau terminal",
    )
    append_effective = _mapping(
        append.get("effective_plateau"),
        label="global-singleton append effective plateau",
    )
    plateau_effective = _mapping(
        plateau.get("effective_plateau"),
        label="global-singleton plateau effective plateau",
    )
    append_insertions = _mapping(
        append.get("insertion_counts"),
        label="global-singleton append insertion counts",
    )
    plateau_insertions = _mapping(
        plateau.get("insertion_counts"),
        label="global-singleton plateau insertion counts",
    )
    derived = _mapping(
        diagnostic.get("derived"),
        label="global-singleton weak-weak derived comparison",
    )
    work_ratio = float(
        derived["terminal_s_alg_ratio_plateau_over_append"]
    )
    n2q_increase = (
        float(plateau_effective["N2q"])
        / max(float(append_effective["N2q"]), 1.0)
        - 1.0
    ) * 100.0
    d2q_increase = (
        float(plateau_effective["D2q"])
        / max(float(append_effective["D2q"]), 1.0)
        - 1.0
    ) * 100.0
    dc_increase = (
        float(plateau_effective["Dc"])
        / max(float(append_effective["Dc"]), 1.0)
        - 1.0
    ) * 100.0

    def tuple_tex(observation: Mapping[str, Any]) -> str:
        return (
            "$("
            f"{int(observation['N2q'])},"
            f"{int(observation['D2q'])},"
            f"{int(observation['Dc'])}"
            ")$"
        )

    return rf"""
\begin{{center}}
{{\large\bfseries Separate weak--weak global-singleton insertion diagnostic}}\\[-0.1ex]
{{\fontsize{{7.25}}{{8.2}}\selectfont Stationary gradients and all-phase cost in both arms; insertion policy is the sole changed axis.}}
\end{{center}}
\vspace{{0.25ex}}
\fcolorbox{{black!35}}{{black!2}}{{%
\begin{{minipage}}{{0.975\textwidth}}
\fontsize{{6.35}}{{7.2}}\selectfont
\textbf{{Matched problem.}} Hubbard--Holstein $L=2$, weak--weak,
$n_{{\rm ph}}=3$ (8 qubits); one global guarded single-Pauli-word candidate
supply; Powell-200; seed 7; fixed 50-round horizon.
\par
\textbf{{Route contract.}} Both RA arms use stationary-source gradients,
all-phase resource weighting (including the Phase-I cost term), retained
singleton reduction, and the same source-locked package. The package
cross-arm audit passed with \texttt{{insertion\_policy}} as the only allowed
axis. ``Append'' here means RA append placement on the commutation-reduced
scaffold; it is not conventional Append-ADAPT.
\end{{minipage}}}}
\vspace{{0.15ex}}
\begin{{center}}
\includegraphics[width=0.985\textwidth,height=3.55in,keepaspectratio]{{{_tex_escape(plot_pdf.name)}}}
\end{{center}}
\vspace{{-0.55ex}}
{{\fontsize{{6.55}}{{7.35}}\selectfont
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lrrrrl}}
\toprule
RA insertion placement & $k_{{\rm eff}}$ & $|\Delta E_{{50}}|$ &
$S_{{\rm alg,50}}$ & $(N_{{2q}},D_{{2q}},D_c)_{{k_{{\rm eff}}}}$ &
placements \\
\midrule
Append, commutation-reduced &
{int(append_effective["k"])} &
{_format_error_tex(append_terminal["error"])} &
${_format_s_alg_tex(int(append_terminal["S_alg"]))}$ &
{tuple_tex(append_effective)} &
{int(append_insertions["interior_count"])} interior /
{int(append_insertions["append_count"])} append \\
Plateau, commutation-reduced &
{int(plateau_effective["k"])} &
{_format_error_tex(plateau_terminal["error"])} &
${_format_s_alg_tex(int(plateau_terminal["S_alg"]))}$ &
{tuple_tex(plateau_effective)} &
{int(plateau_insertions["interior_count"])} interior /
{int(plateau_insertions["append_count"])} append \\
\bottomrule
\end{{tabular*}}
}}
\vspace{{0.35ex}}
{{\fontsize{{6.25}}{{7.15}}\selectfont
\textbf{{Direct outcome.}} Both arms finish at the double-precision error
floor, so the $2.9\times10^{{-15}}$ difference between their terminal errors
is not a meaningful energy advantage. Plateau placement uses
{work_ratio:.3f}$\times$ the terminal algorithmic work and, at each route's
effective plateau, is slightly larger:
$N_{{2q}}$ +{n2q_increase:.1f}\%, $D_{{2q}}$ +{d2q_increase:.1f}\%, and
$D_c$ +{dc_increase:.1f}\%.
\par
\textbf{{Mechanism and boundary.}} Plateau placement was genuinely exercised:
{int(plateau_insertions["interior_count"])} interior insertions, first at
round {int(plateau_insertions["first_interior_round"])}. This is one
weak--weak cell, stopped at a prescribed horizon rather than natural
convergence. It is a separate all-phase-cost diagnostic, not a pages 1--2
stationary-core observation and not adopted Paper-I evidence.
}}
"""


def _write_tex(
    *,
    output_dir: Path,
    document_stem: str,
    macro_plot_pdf: Path,
    singleton_plot_pdf: Path,
    cells: Sequence[Mapping[str, Any]],
    parameter_manifest: Mapping[str, Any],
    pending: bool,
    partial: bool = False,
    supplemental_page_tex: Sequence[str] = (),
) -> Path:
    tex = output_dir / f"{document_stem}.tex"
    macro_page = _report_page_tex(
        cells=cells,
        representation_key="macro",
        plot_pdf=macro_plot_pdf,
        manifest=parameter_manifest,
        include_manifest=True,
        pending=pending,
        partial=partial,
    )
    singleton_page = _report_page_tex(
        cells=cells,
        representation_key="singleton",
        plot_pdf=singleton_plot_pdf,
        manifest=parameter_manifest,
        include_manifest=False,
        pending=pending,
        partial=partial,
    )
    supplemental_pages = "".join(
        "\n\\clearpage\n" + page
        for page in supplemental_page_tex
    )
    body = rf"""\documentclass[letterpaper]{{article}}
\usepackage[margin=0.24in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{url}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\tabcolsep}}{{1.8pt}}
\begin{{document}}
{macro_page}
\clearpage
{singleton_page}
{supplemental_pages}
\end{{document}}
"""
    tex.write_text(body, encoding="utf-8")
    return tex


def _compile_tex(tex: Path) -> Path:
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    build_dir = REPO_ROOT / "tmp" / "pdfs" / tex.stem
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
        raise RuntimeError("latexmk or pdflatex is required to build the report")
    completed = subprocess.run(
        command,
        cwd=tex.parent,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "FORCE_SOURCE_DATE": "1",
            "SOURCE_DATE_EPOCH": "1785196800",
            "TZ": "UTC",
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "LaTeX report build failed:\n"
            + completed.stdout[-4000:]
            + completed.stderr[-4000:]
        )
    compiled_pdf = build_dir / f"{tex.stem}.pdf"
    if not compiled_pdf.is_file():
        raise RuntimeError("LaTeX completed without producing the report PDF")
    pdf = tex.with_suffix(".pdf")
    shutil.copy2(compiled_pdf, pdf)
    return pdf


def _package_sources() -> dict[str, Any]:
    _configure_package_dir(PACKAGE_DIR)
    manifest, manifest_sha = _verified_object(
        PACKAGE_DIR / "package_manifest.json", label="package manifest"
    )
    plan, plan_sha = _verified_object(
        PACKAGE_DIR / "execution_plan.json", label="execution plan"
    )
    source, source_sha = _verified_object(
        PACKAGE_DIR / "source_archive_manifest.json",
        label="source archive manifest",
    )
    return {
        "package_manifest": {
            "path": str(PACKAGE_DIR / "package_manifest.json"),
            "sha256": manifest_sha,
            "file_sha256": _sha256_file(PACKAGE_DIR / "package_manifest.json"),
        },
        "execution_plan": {
            "path": str(PACKAGE_DIR / "execution_plan.json"),
            "sha256": plan_sha,
            "file_sha256": _sha256_file(PACKAGE_DIR / "execution_plan.json"),
        },
        "source_archive_manifest": {
            "path": str(PACKAGE_DIR / "source_archive_manifest.json"),
            "sha256": source_sha,
            "file_sha256": _sha256_file(
                PACKAGE_DIR / "source_archive_manifest.json"
            ),
        },
        "source_archive_sha256": source["archive"]["sha256"],
        "remote_image_sha256": manifest["remote_image"]["sha256"],
        "execution_count": plan["direct_execution_count"],
        "core_materialization_id": CORE_MATERIALIZATION_ID,
        "core_final_receipt": manifest["core_final_receipt"],
    }


def _protocol_for_job(job: Mapping[str, Any]) -> dict[str, Any]:
    binding = _mapping(job.get("protocol"), label="job protocol binding")
    relative = PurePosixPath(str(binding.get("path", "")))
    if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
        raise ReportInputError("job protocol path is unsafe")
    path = REPO_ROOT.joinpath(*relative.parts)
    if (
        not path.is_file()
        or path.is_symlink()
        or _sha256_file(path) != binding.get("sha256")
        or path.stat().st_size != binding.get("size_bytes")
    ):
        raise ReportInputError("job protocol file binding drifted")
    protocol, digest = _verified_object(path, label="job protocol")
    if digest != binding.get("canonical_sha256"):
        raise ReportInputError("job protocol canonical digest drifted")
    return protocol


def _parameter_manifest() -> dict[str, Any]:
    jobs = _expected_jobs()
    regimes: dict[str, dict[str, Any]] = {}
    shared: dict[str, Any] | None = None
    route_ids: set[str] = set()
    representations: set[str] = set()
    for execution_id, job in sorted(jobs.items()):
        protocol = _protocol_for_job(job)
        problem = dict(
            _mapping(
                protocol.get("problem"),
                label=f"{execution_id} problem",
            )
        )
        regime = str(job["regime_id"])
        normalized_problem = {
            "regime_id": regime,
            "label": REGIME_LABELS[regime][1].replace("--", "-"),
            "problem_request_sha256": problem["problem_request_sha256"],
            "t": problem["t"],
            "u": problem["u"],
            "dv": problem["dv"],
            "omega0": problem["omega0"],
            "g_ep": problem["g_ep"],
            "n_ph_max": problem["n_ph_max"],
            "v_nn": problem["v_nn"],
            "t_prime": problem["t_prime"],
            "total_qubits": problem["total_qubits"],
        }
        previous = regimes.get(regime)
        if previous is not None and previous != normalized_problem:
            raise ReportInputError(
                f"{regime}: protocol problem settings disagree"
            )
        regimes[regime] = normalized_problem
        row_shared = {
            "model_family": "Hubbard-Holstein",
            "problem_key": problem["problem_key"],
            "num_sites": problem["num_sites"],
            "boundary": problem["boundary"],
            "boson_encoding": problem["boson_encoding"],
            "ordering": problem["ordering"],
            "include_zero_point": problem["include_zero_point"],
            "sector_label": problem["sector_label"],
            "exact_target_label": problem["exact_target_label"],
            "drive_enabled": False,
            "optimizer": str(protocol["optimizer"]).lower(),
            "optimizer_maxiter": protocol["optimizer_maxiter"],
            "horizon": protocol["horizon"],
            "seeds": dict(protocol["seeds"]),
            "active_gradient_policy": protocol["active_gradient_policy"],
            "resource_weighting_scope": protocol[
                "resource_weighting_scope"
            ],
        }
        if shared is not None and shared != row_shared:
            raise ReportInputError("shared protocol settings disagree")
        shared = row_shared
        route_ids.add(str(job["route_id"]))
        representations.add(str(job["candidate_representation"]))
    if shared is None or set(regimes) != set(REGIME_ORDER):
        raise ReportInputError("parameter manifest lacks the six regimes")
    return {
        "schema": (
            "paper_i_ra_adapt_stationary_core_parameter_manifest_v1"
        ),
        **shared,
        "ansatz_representations": sorted(representations),
        "route_ids": sorted(route_ids),
        "methods": [
            {
                "method": method,
                "label": METHODS[method]["label"],
            }
            for method in METHOD_ORDER
        ],
        "regimes": [regimes[regime] for regime in REGIME_ORDER],
        "reference_definition": (
            "exact_ground_state_energy_at_identical_n_ph_max"
        ),
        "error_metric": "same_cutoff_absolute_energy_error",
        "package_provenance": _package_sources(),
    }


def _terminal_cost_policy() -> dict[str, Any]:
    return {
        "controller_round": 50,
        "tuple_fields": list(PAPER_I_QISKIT_COST_TUPLE_FIELDS),
        "tuple_latex": PAPER_I_QISKIT_COST_TUPLE_LATEX,
        "compile_convention": "table_i_basis_gate_transpile_v1",
        "common_qiskit_path": (
            "method_specific_authenticated_terminal_prefix_to_"
            "PaperIPrefixCompileInput_to_shared_locked_compiler_v1"
        ),
        "ra_adapt_input": "authenticated_terminal_replay_checkpoint",
        "append_input": (
            "authenticated_signed_terminal_checkpoint_plus_"
            "protocol_locked_executable_pool"
        ),
        "append_serialized_terminal_role": (
            "mandatory_exact_recompile_cross_check"
        ),
        "qiskit_coordinates_independent_of_s_alg": True,
        "fourth_coordinate": {
            "field": "W1q",
            "source_field": (
                "qiskit_pretranspile_pauli_1q_work_total"
            ),
            "semantics": (
                "qiskit_emitted_h_plus_s_plus_sdg_plus_rz_before_"
                "transpilation_excluding_reference_state_preparation"
            ),
            "basis_change_only_field": "B1q",
        },
        "fifth_coordinate": {
            "field": "S_alg",
            "semantics": "closed_logical_scalar_estimator_occurrences",
            "display_notation": "X.YeZ_two_significant_digits",
        },
        "plot_overlay": {
            "scope": "every_completed_curve",
            "controller_round": 50,
            "tuple_fields": list(PAPER_I_QISKIT_COST_TUPLE_FIELDS),
            "method_identity": "legend_matched_color_and_symbol",
            "pending_cells_annotated": False,
        },
        "effective_plateau_resources_used_as_terminal": False,
    }


def build_pending_preview(
    *,
    output_dir: Path | None = None,
    package_dir: Path | None = None,
) -> tuple[Path, Path]:
    _configure_package_dir(PACKAGE_DIR if package_dir is None else package_dir)
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = _pending_cells()
    parameter_manifest = _parameter_manifest()
    macro_png = output_dir / f"{STEM}_pending_macro_master.png"
    singleton_png = output_dir / f"{STEM}_pending_singleton_master.png"
    macro_plot_pdf = output_dir / f"{STEM}_pending_macro_plots.pdf"
    singleton_plot_pdf = output_dir / f"{STEM}_pending_singleton_plots.pdf"
    _render_plot_grid(
        cells=cells,
        representation_key="macro",
        vector_output=macro_plot_pdf,
        png_output=macro_png,
        pending=True,
    )
    _render_plot_grid(
        cells=cells,
        representation_key="singleton",
        vector_output=singleton_plot_pdf,
        png_output=singleton_png,
        pending=True,
    )
    document_stem = f"{STEM}_pending_preview"
    tex = _write_tex(
        output_dir=output_dir,
        document_stem=document_stem,
        macro_plot_pdf=macro_plot_pdf,
        singleton_plot_pdf=singleton_plot_pdf,
        cells=cells,
        parameter_manifest=parameter_manifest,
        pending=True,
    )
    pdf = _compile_tex(tex)
    pending_path = output_dir / f"{STEM}_pending.json"
    pending_payload = {
        "schema": "paper_i_ra_adapt_stationary_core_master_pending_v1",
        "package_id": PACKAGE_ID,
        "status": "awaiting_explicit_selection_of_48_validated_attempts",
        "not_paper_evidence": True,
        "canonical_results_pdf_emitted": False,
        "parameter_manifest": parameter_manifest,
        "terminal_cost_policy": _terminal_cost_policy(),
        "missing_execution_ids": sorted(
            str(row["execution_id"]) for row in cells
        ),
        "layout": {
            "page_count": 2,
            "page_1": "macro_generator_v1",
            "page_2": "single_pauli_word_v1",
            "regime_count_per_page": 6,
            "route_count_per_regime": 4,
            "terminal_rows_per_page": 24,
        },
        "package_sources": _package_sources(),
        "outputs": {
            "preview_pdf": {
                "path": str(pdf),
                "sha256": _sha256_file(pdf),
            },
            "macro_png": {
                "path": str(macro_png),
                "sha256": _sha256_file(macro_png),
            },
            "macro_plot_pdf": {
                "path": str(macro_plot_pdf),
                "sha256": _sha256_file(macro_plot_pdf),
            },
            "singleton_png": {
                "path": str(singleton_png),
                "sha256": _sha256_file(singleton_png),
            },
            "singleton_plot_pdf": {
                "path": str(singleton_plot_pdf),
                "sha256": _sha256_file(singleton_plot_pdf),
            },
            "tex": {"path": str(tex), "sha256": _sha256_file(tex)},
        },
    }
    pending_path.write_text(
        json.dumps(pending_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pdf, pending_path


def build_partial_progress(
    *,
    validation_path: Path,
    fetched_dir: Path,
    output_dir: Path | None = None,
    package_dir: Path | None = None,
    terminal_qiskit_compiler: Callable[[Any], Any] | None = None,
) -> tuple[Path, Path]:
    """Build a non-evidentiary two-page report from a validated subset."""

    _configure_package_dir(PACKAGE_DIR if package_dir is None else package_dir)
    if output_dir is None:
        output_dir = OUTPUT_DIR
    included_cells, validation_sources = load_partial_cells(
        validation_path=validation_path,
        fetched_dir=fetched_dir,
        terminal_qiskit_compiler=terminal_qiskit_compiler,
    )
    cells = _merge_partial_with_pending(included_cells)
    included_execution_ids = sorted(
        str(row["execution_id"]) for row in included_cells
    )
    missing_execution_ids = sorted(
        str(row["execution_id"])
        for row in cells
        if not row.get("points")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_manifest = _parameter_manifest()
    macro_png = output_dir / f"{STEM}_partial_macro_master.png"
    singleton_png = output_dir / f"{STEM}_partial_singleton_master.png"
    macro_plot_pdf = output_dir / f"{STEM}_partial_macro_plots.pdf"
    singleton_plot_pdf = (
        output_dir / f"{STEM}_partial_singleton_plots.pdf"
    )
    _render_plot_grid(
        cells=cells,
        representation_key="macro",
        vector_output=macro_plot_pdf,
        png_output=macro_png,
        pending=False,
        partial=True,
    )
    _render_plot_grid(
        cells=cells,
        representation_key="singleton",
        vector_output=singleton_plot_pdf,
        png_output=singleton_png,
        pending=False,
        partial=True,
    )
    document_stem = f"{STEM}_partial_progress"
    tex = _write_tex(
        output_dir=output_dir,
        document_stem=document_stem,
        macro_plot_pdf=macro_plot_pdf,
        singleton_plot_pdf=singleton_plot_pdf,
        cells=cells,
        parameter_manifest=parameter_manifest,
        pending=False,
        partial=True,
    )
    pdf = _compile_tex(tex)
    provenance_path = (
        output_dir / f"{STEM}_partial_progress_provenance.json"
    )
    provenance = {
        "schema": (
            "paper_i_ra_adapt_stationary_core_master_partial_progress_v1"
        ),
        "package_id": PACKAGE_ID,
        "status": "partial_validated_results_non_evidentiary",
        "partial_progress": True,
        "not_paper_evidence": True,
        "paper_evidence_adopted": False,
        "canonical_results_pdf_emitted": False,
        "final_selection_consumed": False,
        "included_count": len(included_execution_ids),
        "pending_count": len(missing_execution_ids),
        "included_execution_ids": included_execution_ids,
        "missing_execution_ids": missing_execution_ids,
        "metric": "same_cutoff_absolute_energy_error",
        "display_rounds": list(range(0, 51)),
        "parameter_manifest": parameter_manifest,
        "terminal_cost_policy": _terminal_cost_policy(),
        "layout": {
            "page_count": 2,
            "page_1": "macro_generator_v1",
            "page_2": "single_pauli_word_v1",
            "regime_count_per_page": 6,
            "route_count_per_regime": 4,
            "terminal_rows_per_page": 24,
        },
        "package_sources": _package_sources(),
        **validation_sources,
        "limitations": [
            (
                "This is a partial progress diagnostic, not a complete "
                "48-cell result and not Paper-I evidence."
            ),
            (
                "Only execution ids with exactly one passed attempt in the "
                "validated receipt are included; the report never chooses "
                "among successful retries."
            ),
            (
                "Every included cell rechecks the exact package job/source "
                "binding, G1-G13 closure, and same-cutoff ED reference."
            ),
        ],
        "outputs": {
            "partial_progress_pdf": {
                "path": str(pdf),
                "sha256": _sha256_file(pdf),
            },
            "macro_png": {
                "path": str(macro_png),
                "sha256": _sha256_file(macro_png),
            },
            "macro_plot_pdf": {
                "path": str(macro_plot_pdf),
                "sha256": _sha256_file(macro_plot_pdf),
            },
            "singleton_png": {
                "path": str(singleton_png),
                "sha256": _sha256_file(singleton_png),
            },
            "singleton_plot_pdf": {
                "path": str(singleton_plot_pdf),
                "sha256": _sha256_file(singleton_plot_pdf),
            },
            "tex": {"path": str(tex), "sha256": _sha256_file(tex)},
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pdf, provenance_path


def build_cross_revision_partial_progress(
    *,
    source_specs: Sequence[Mapping[str, Any]],
    recovery_adapter_paths: Sequence[Path] = (),
    diagnostic_local_always_prefixes: Sequence[Mapping[str, Any]] = (),
    output_dir: Path | None = None,
    terminal_qiskit_compiler: Callable[[Any], Any] | None = None,
    diagnostic_qiskit_plateau_run_dir: Path | None = None,
    diagnostic_qiskit_plateau_log: Path | None = None,
    diagnostic_qiskit_always_run_dir: Path | None = None,
    global_singleton_weak_weak_adapter_path: Path | None = None,
    diagnostic_qiskit_singleton_run_dir: Path | None = None,
    diagnostic_cumulative_plateau_macro_run_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Build one evolving report from disjoint explicit package sources."""

    if (diagnostic_qiskit_plateau_run_dir is None) != (
        diagnostic_qiskit_plateau_log is None
    ):
        raise ReportInputError(
            "the Qiskit-plateau diagnostic requires both its run directory "
            "and remote-runner log"
        )
    if (
        diagnostic_qiskit_always_run_dir is not None
        and diagnostic_qiskit_plateau_run_dir is None
    ):
        raise ReportInputError(
            "the Qiskit-always diagnostic extends the plateau comparison page"
        )
    if output_dir is None:
        output_dir = CROSS_REVISION_OUTPUT_DIR
    included_cells, source_provenance, parameter_manifest = (
        load_cross_revision_partial_cells(
            source_specs=source_specs,
            recovery_adapter_paths=recovery_adapter_paths,
            terminal_qiskit_compiler=terminal_qiskit_compiler,
        )
    )
    local_paused_sources: list[dict[str, Any]] = []
    if diagnostic_local_always_prefixes:
        expected_jobs = _expected_jobs()
        exact_by_regime = {
            str(row["regime"]): _finite(
                row.get("exact_same_cutoff_energy"),
                label=f"{row['execution_id']} exact same-cutoff energy",
            )
            for row in included_cells
            if (
                row.get("method") == "append"
                and row.get("representation") == "macro_generator_v1"
            )
        }
        seen_ids = {str(row["execution_id"]) for row in included_cells}
        for index, raw in enumerate(
            diagnostic_local_always_prefixes,
            start=1,
        ):
            spec = _mapping(
                raw,
                label=f"local paused-prefix specification {index}",
            )
            job_path = Path(str(spec.get("job_path", ""))).resolve()
            job = _load_object(
                job_path,
                label=f"local paused-prefix job {index}",
            )
            regime = str(job.get("regime_id", ""))
            if regime not in exact_by_regime:
                raise ReportInputError(
                    f"local paused-prefix {index} lacks a matched Append source"
                )
            cell, source = _load_local_paused_always_prefix(
                job_path=job_path,
                checkpoint_path=Path(
                    str(spec.get("checkpoint_path", ""))
                ).resolve(),
                log_path=Path(str(spec.get("log_path", ""))).resolve(),
                expected_jobs=expected_jobs,
                exact_same_cutoff_energy=exact_by_regime[regime],
            )
            execution_id = str(cell["execution_id"])
            if execution_id in seen_ids:
                raise ReportInputError(
                    "local paused-prefix overlaps an included cell: "
                    + execution_id
                )
            seen_ids.add(execution_id)
            included_cells.append(cell)
            local_paused_sources.append(source)

        package_provenance = dict(
            _mapping(
                parameter_manifest.get("package_provenance"),
                label="cross-revision package provenance",
            )
        )
        package_provenance["package_ids"] = sorted(
            {
                *(
                    str(value)
                    for value in _sequence(
                        package_provenance.get("package_ids"),
                        label="cross-revision package ids",
                    )
                ),
                *(str(row["package_id"]) for row in local_paused_sources),
            }
        )
        package_provenance["local_paused_prefix_count"] = len(
            local_paused_sources
        )
        package_provenance["local_paused_prefixes"] = local_paused_sources
        parameter_manifest = {
            **parameter_manifest,
            "package_provenance": package_provenance,
        }
        source_provenance = {
            **source_provenance,
            "source_policy": (
                str(source_provenance["source_policy"])
                + "_plus_authenticated_local_paused_prefixes_v1"
            ),
            "included_sources": [
                *list(
                    _sequence(
                        source_provenance.get("included_sources"),
                        label="cross-revision included sources",
                    )
                ),
                *local_paused_sources,
            ],
            "local_paused_prefix_sources": local_paused_sources,
        }
    cells = _merge_partial_with_pending(included_cells)
    included_execution_ids = sorted(
        str(row["execution_id"]) for row in included_cells
    )
    missing_execution_ids = sorted(
        str(row["execution_id"])
        for row in cells
        if not row.get("points")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    macro_png = output_dir / f"{CROSS_REVISION_STEM}_macro_master.png"
    singleton_png = (
        output_dir / f"{CROSS_REVISION_STEM}_singleton_master.png"
    )
    macro_plot_pdf = (
        output_dir / f"{CROSS_REVISION_STEM}_macro_plots.pdf"
    )
    singleton_plot_pdf = (
        output_dir / f"{CROSS_REVISION_STEM}_singleton_plots.pdf"
    )
    diagnostic: dict[str, Any] | None = None
    diagnostic_page_tex: str | None = None
    diagnostic_plot_pdf: Path | None = None
    diagnostic_png: Path | None = None
    append_cell: Mapping[str, Any] | None = None
    always_diagnostic: dict[str, Any] | None = None
    proxy_comparison: dict[str, Any] | None = None
    supplemental_pages: list[str] = []
    if diagnostic_qiskit_plateau_run_dir is not None:
        append_matches = [
            row
            for row in included_cells
            if (
                row.get("execution_id")
                == "core__strong_weak_u8__nph3__append_macro"
                and row.get("method") == "append"
            )
        ]
        if len(append_matches) != 1:
            raise ReportInputError(
                "the diagnostic page requires exactly one validated matched "
                "strong-weak macro Append cell"
            )
        append_cell = append_matches[0]
        if diagnostic_qiskit_always_run_dir is None:
            raise ReportInputError(
                "the current diagnostic page requires the completed "
                "Qiskit-always run"
            )
        proxy_matches = {
            str(row.get("method")): row
            for row in included_cells
            if row.get("execution_id")
            in {
                "core__strong_weak_u8__nph3__ra_macro_append_only",
                "core__strong_weak_u8__nph3__ra_macro_plateau",
            }
        }
        if set(proxy_matches) != {"no_insertion", "plateau"}:
            raise ReportInputError(
                "the diagnostic page requires page-1 proxy-ranked "
                "strong-weak macro plateau and no-insertion cells"
            )
        proxy_comparison = {
            method: dict(
                _mapping(
                    proxy_matches[method].get("fixed_iteration_qiskit"),
                    label=f"proxy-ranked {method} fixed iteration",
                )
            )
            for method in ("plateau", "no_insertion")
        }
        diagnostic = _load_qiskit_plateau_macro_diagnostic(
            run_dir=diagnostic_qiskit_plateau_run_dir,
            log_path=diagnostic_qiskit_plateau_log,
            append_cell=append_cell,
            compiler=terminal_qiskit_compiler,
        )
        always_diagnostic = _load_qiskit_always_macro_diagnostic(
            run_dir=diagnostic_qiskit_always_run_dir,
            exact_energy=_finite(
                append_cell.get("exact_same_cutoff_energy"),
                label="matched Append same-cutoff energy",
            ),
            compiler=terminal_qiskit_compiler,
        )
        diagnostic_plot_pdf = output_dir / (
            f"{CROSS_REVISION_STEM}_"
            "qiskit_plateau_vs_append_macro_plot.pdf"
        )
        diagnostic_png = output_dir / (
            f"{CROSS_REVISION_STEM}_"
            "qiskit_plateau_vs_append_macro_plot.png"
        )
        _render_qiskit_plateau_append_comparison(
            diagnostic=diagnostic,
            always_diagnostic=always_diagnostic,
            append_cell=append_cell,
            vector_output=diagnostic_plot_pdf,
            png_output=diagnostic_png,
        )
        diagnostic_page_tex = _qiskit_plateau_comparison_page_tex(
            diagnostic=diagnostic,
            always_diagnostic=always_diagnostic,
            append_cell=append_cell,
            proxy_comparison=proxy_comparison,
            plot_pdf=diagnostic_plot_pdf,
        )
        supplemental_pages.append(diagnostic_page_tex)
    global_singleton_diagnostic: dict[str, Any] | None = None
    global_singleton_plot_pdf: Path | None = None
    global_singleton_png: Path | None = None
    if global_singleton_weak_weak_adapter_path is not None:
        global_singleton_diagnostic = (
            _load_global_singleton_weak_weak_diagnostic(
                global_singleton_weak_weak_adapter_path
            )
        )
        global_singleton_plot_pdf = output_dir / (
            f"{CROSS_REVISION_STEM}_"
            "global_singleton_weak_weak_append_vs_plateau_plot.pdf"
        )
        global_singleton_png = output_dir / (
            f"{CROSS_REVISION_STEM}_"
            "global_singleton_weak_weak_append_vs_plateau_plot.png"
        )
        _render_global_singleton_weak_weak_comparison(
            diagnostic=global_singleton_diagnostic,
            vector_output=global_singleton_plot_pdf,
            png_output=global_singleton_png,
        )
        supplemental_pages.append(
            _global_singleton_weak_weak_page_tex(
                diagnostic=global_singleton_diagnostic,
                plot_pdf=global_singleton_plot_pdf,
            )
        )
    singleton_round33_diagnostic: dict[str, Any] | None = None
    singleton_round33_plot_pdf: Path | None = None
    singleton_round33_png: Path | None = None
    if diagnostic_qiskit_singleton_run_dir is not None:
        matched_cells = {
            str(row.get("execution_id")): row
            for row in included_cells
            if row.get("execution_id") in MATCHED_SINGLETON_EXECUTION_IDS
        }
        if set(matched_cells) != set(MATCHED_SINGLETON_EXECUTION_IDS):
            raise ReportInputError(
                "the singleton round-33 page requires the matched staged "
                "RA and conventional Append cells"
            )
        singleton_round33_diagnostic = (
            _load_qiskit_singleton_round33_diagnostic(
                run_dir=diagnostic_qiskit_singleton_run_dir,
                ra_cell=matched_cells[
                    "core__strong_strong_u8__nph7__ra_singleton_plateau"
                ],
                append_cell=matched_cells[
                    "core__strong_strong_u8__nph7__append_singleton"
                ],
                compiler=terminal_qiskit_compiler,
            )
        )
        singleton_round33_plot_pdf = output_dir / (
            f"{CROSS_REVISION_STEM}_"
            "strong_strong_singleton_qiskit_round33_comparison_plot.pdf"
        )
        singleton_round33_png = output_dir / (
            f"{CROSS_REVISION_STEM}_"
            "strong_strong_singleton_qiskit_round33_comparison_plot.png"
        )
        _render_qiskit_singleton_round33_comparison(
            diagnostic=singleton_round33_diagnostic,
            vector_output=singleton_round33_plot_pdf,
            png_output=singleton_round33_png,
        )
        supplemental_pages.append(
            _qiskit_singleton_round33_page_tex(
                diagnostic=singleton_round33_diagnostic,
                plot_pdf=singleton_round33_plot_pdf,
            )
        )
    cumulative_plateau_macro_diagnostic: dict[str, Any] | None = None
    if diagnostic_cumulative_plateau_macro_run_dir is not None:
        matched = {
            str(row.get("method")): row
            for row in included_cells
            if row.get("execution_id")
            == CUMULATIVE_PLATEAU_MACRO_EXECUTION_ID
        }
        append_matches = [
            row
            for row in included_cells
            if row.get("execution_id")
            == "core__intermediate_strong__nph7__append_macro"
        ]
        if set(matched) != {"plateau"} or len(append_matches) != 1:
            raise ReportInputError(
                "the cumulative-plateau overlay requires one matched "
                "stationary plateau and one Append cell"
            )
        cumulative_plateau_macro_diagnostic = (
            _load_cumulative_plateau_macro_diagnostic(
                run_dir=diagnostic_cumulative_plateau_macro_run_dir,
                plateau_cell=matched["plateau"],
                append_cell=append_matches[0],
            )
        )
    _render_plot_grid(
        cells=cells,
        representation_key="macro",
        vector_output=macro_plot_pdf,
        png_output=macro_png,
        pending=False,
        partial=True,
        diagnostic_overlays=(
            ()
            if cumulative_plateau_macro_diagnostic is None
            else (cumulative_plateau_macro_diagnostic,)
        ),
    )
    _render_plot_grid(
        cells=cells,
        representation_key="singleton",
        vector_output=singleton_plot_pdf,
        png_output=singleton_png,
        pending=False,
        partial=True,
    )
    tex = _write_tex(
        output_dir=output_dir,
        document_stem=CROSS_REVISION_STEM,
        macro_plot_pdf=macro_plot_pdf,
        singleton_plot_pdf=singleton_plot_pdf,
        cells=cells,
        parameter_manifest=parameter_manifest,
        pending=False,
        partial=True,
        supplemental_page_tex=tuple(supplemental_pages),
    )
    pdf = _compile_tex(tex)
    provenance_path = (
        output_dir / f"{CROSS_REVISION_STEM}_provenance.json"
    )
    package_provenance = _mapping(
        parameter_manifest.get("package_provenance"),
        label="cross-revision package provenance",
    )
    package_ids = list(
        _sequence(
            package_provenance.get("package_ids"),
            label="cross-revision package ids",
        )
    )
    recovery_count = _integer(
        package_provenance.get("recovery_cell_count", 0),
        label="cross-revision recovery cell count",
    )
    local_paused_count = _integer(
        package_provenance.get("local_paused_prefix_count", 0),
        label="cross-revision local paused-prefix count",
    )
    recovery_counts: dict[str, int] = {
        RECOVERY_CROSS_CAMPAIGN_CLASS: 0,
        RECOVERY_G5_UNEXERCISED_CLASS: 0,
    }
    for raw in _sequence(
        package_provenance.get("recovery_adapters", ()),
        label="cross-revision recovery adapters",
    ):
        counts = _mapping(
            _mapping(
                raw,
                label="cross-revision recovery adapter",
            ).get("recovery_counts"),
            label="cross-revision recovery counts",
        )
        for recovery_class in recovery_counts:
            recovery_counts[recovery_class] += _integer(
                counts.get(recovery_class, 0),
                label=f"cross-revision {recovery_class} count",
            )
    if sum(recovery_counts.values()) != recovery_count:
        raise ReportInputError(
            "cross-revision recovery class counts do not close"
        )
    ordinary_validated_count = (
        len(included_execution_ids) - recovery_count - local_paused_count
    )
    if ordinary_validated_count < 0:
        raise ReportInputError(
            "cross-revision recovery count exceeds included count"
        )
    layout = {
        "page_count": 2 + len(supplemental_pages),
        "page_1": "macro_generator_v1",
        "page_2": "single_pauli_word_v1",
        "regime_count_per_page": 6,
        "route_count_per_regime": 4,
        "terminal_rows_per_page": 24,
        "page_1_diagnostic_overlay_count": (
            0 if cumulative_plateau_macro_diagnostic is None else 1
        ),
    }
    next_supplemental_page = 3
    if diagnostic is not None:
        layout[f"page_{next_supplemental_page}"] = (
            "strong_weak_macro_qiskit_ranked_insertion_vs_proxy_diagnostic_v2"
        )
        next_supplemental_page += 1
    if global_singleton_diagnostic is not None:
        layout[f"page_{next_supplemental_page}"] = (
            "weak_weak_global_singleton_append_vs_plateau_diagnostic_v1"
        )
        next_supplemental_page += 1
    if singleton_round33_diagnostic is not None:
        layout[f"page_{next_supplemental_page}"] = (
            "strong_strong_singleton_matched_round33_qiskit_comparison_v1"
        )
    limitations = [
        (
            "This is an evolving cross-revision progress diagnostic, "
            "not a complete 48-cell result and not Paper-I evidence."
        ),
        (
            "Append and RA cells are admitted only from their explicitly "
            "declared source families; overlapping passed cells fail "
            "closed instead of being selected automatically."
        ),
    ]
    if recovery_count:
        limitations.extend(
            (
                (
                    f"{ordinary_validated_count} ordinary cells come from "
                    "exactly one passed validated attempt and recheck their "
                    "package job/source binding, G1-G13 closure, and "
                    "same-cutoff ED reference."
                ),
                (
                    f"{recovery_counts[RECOVERY_CROSS_CAMPAIGN_CLASS]} "
                    "RA-always observations come from passed factorial "
                    "attempts mapped through an explicit cross-campaign "
                    "science-equivalence projection. They match this "
                    "report's stationary, late-resource-weighting "
                    "(Phase-I cost-off), commutation-reduced baseline, but "
                    "remain diagnostic rather than adopted paper evidence."
                ),
                (
                    f"{recovery_counts[RECOVERY_G5_UNEXERCISED_CLASS]} "
                    "completed plateau trajectories are retained failed "
                    "attempts: G1-G4 and G6-G13 pass, but G5 fails because "
                    "no interior insertion position was scored. They are "
                    "shown as G5-unexercised diagnostics and are not valid "
                    "plateau evidence."
                ),
            )
        )
    else:
        limitations.extend(
            (
                (
                    "Only attempts whose validation status is passed are "
                    "included. Failed and retained failed attempts never "
                    "plot on the 48-cell master pages."
                ),
                (
                    "Every included cell rechecks its own package job/source "
                    "binding, G1-G13 closure, and same-cutoff ED reference."
                ),
            )
        )
    if local_paused_count:
        limitations.append(
            f"{local_paused_count} RA-always curves are deliberately stopped "
            "local accepted prefixes. Their checkpoint and progress-log "
            "histories agree exactly, but they are incomplete, have no "
            "round-50 compiled tuple, and remain diagnostic only."
        )
    if cumulative_plateau_macro_diagnostic is not None:
        limitations.append(
            "Page 1 overlays one authenticated 20-round intermediate--strong "
            "macro cumulative-relative plateau diagnostic. It is not an "
            "additional 48-cell result, has no round-50 cost tuple, and is "
            "not adopted Paper-I evidence."
        )
    diagnostic_provenance: dict[str, Any] | None = None
    diagnostic_outputs: dict[str, Any] = {}
    if (
        diagnostic is not None
        and always_diagnostic is not None
        and proxy_comparison is not None
        and append_cell is not None
        and diagnostic_plot_pdf is not None
        and diagnostic_png is not None
    ):
        matched_append_source = next(
            (
                dict(row)
                for row in _sequence(
                    source_provenance.get("included_sources"),
                    label="cross-revision included sources",
                )
                if _mapping(
                    row,
                    label="cross-revision included source",
                ).get("execution_id")
                == "core__strong_weak_u8__nph3__append_macro"
            ),
            None,
        )
        if matched_append_source is None:
            raise ReportInputError(
                "diagnostic provenance lacks the matched Append source"
            )
        diagnostic_provenance = {
            **diagnostic,
            "metric": "same_cutoff_absolute_energy_error",
            "display_rounds": list(range(0, 51)),
            "plotted_point_count": len(diagnostic["points"]),
            "matched_append_curve": {
                "execution_id": str(append_cell["execution_id"]),
                "points": list(append_cell["points"]),
                "plotted_point_count": len(append_cell["points"]),
                "marker": dict(
                    _mapping(
                        append_cell.get("marker"),
                        label="diagnostic Append marker",
                    )
                ),
                "terminal": dict(
                    _mapping(
                        append_cell.get("terminal"),
                        label="diagnostic Append terminal",
                    )
                ),
                "source": matched_append_source,
            },
            "qiskit_ranked_always_curve": always_diagnostic,
            "fixed_iteration_comparison": {
                "controller_round": FIXED_COMPARISON_ROUND,
                "qiskit_ranked_plateau": dict(
                    _mapping(
                        diagnostic.get("fixed_iteration_qiskit"),
                        label="diagnostic plateau fixed iteration",
                    )
                ),
                "qiskit_ranked_always": dict(
                    _mapping(
                        always_diagnostic.get("fixed_iteration_qiskit"),
                        label="diagnostic always fixed iteration",
                    )
                ),
                "append": dict(
                    _mapping(
                        append_cell.get("fixed_iteration_qiskit"),
                        label="diagnostic Append fixed iteration",
                    )
                ),
                "proxy_ranked_plateau": proxy_comparison["plateau"],
                "proxy_ranked_no_insertion": proxy_comparison[
                    "no_insertion"
                ],
                "proxy_ranked_always": {
                    "status": "unavailable",
                    "reason": "page_1_run_pending",
                },
            },
        }
        limitations.extend(
            (
                (
                    "Page 3 admits no additional 48-cell result: it compares "
                    "a local 50-round Qiskit-ranked plateau trajectory and "
                    "a local 13-round Qiskit-ranked always trajectory with "
                    "validated page-1 routes."
                ),
                (
                    "Page 3 is a whole-route diagnostic rather than an "
                    "insertion-only ablation; the RA and Append routes differ "
                    "in selector, trust, whitening, and cost-ranking logic."
                ),
                (
                    "Page 3 recompiles authenticated iteration-10 prefixes "
                    "through the common Paper-I Qiskit compiler. The "
                    "proxy-ranked always route remains unavailable."
                ),
            )
        )
        diagnostic_outputs = {
            "qiskit_plateau_vs_append_plot_pdf": {
                "path": str(diagnostic_plot_pdf),
                "sha256": _sha256_file(diagnostic_plot_pdf),
            },
            "qiskit_plateau_vs_append_plot_png": {
                "path": str(diagnostic_png),
                "sha256": _sha256_file(diagnostic_png),
            },
        }
    global_singleton_provenance: dict[str, Any] | None = None
    global_singleton_outputs: dict[str, Any] = {}
    if (
        global_singleton_diagnostic is not None
        and global_singleton_plot_pdf is not None
        and global_singleton_png is not None
    ):
        global_singleton_provenance = dict(global_singleton_diagnostic)
        limitations.extend(
            (
                (
                    "The global-singleton weak-weak page admits no "
                    "additional 48-cell result. It is a separate matched "
                    "insertion-policy diagnostic with stationary gradients "
                    "and all-phase resource weighting."
                ),
                (
                    "The global-singleton pair is one prescribed 50-round "
                    "weak-weak cell, not a cross-regime result and not "
                    "adopted Paper-I evidence."
                ),
                (
                    "Global-singleton append placement is an RA route on "
                    "the commutation-reduced scaffold; it is not the "
                    "conventional Append-ADAPT comparator on pages 1-2."
                ),
            )
        )
        global_singleton_outputs = {
            "global_singleton_weak_weak_plot_pdf": {
                "path": str(global_singleton_plot_pdf),
                "sha256": _sha256_file(global_singleton_plot_pdf),
            },
            "global_singleton_weak_weak_plot_png": {
                "path": str(global_singleton_png),
                "sha256": _sha256_file(global_singleton_png),
            },
        }
    singleton_round33_provenance: dict[str, Any] | None = None
    singleton_round33_outputs: dict[str, Any] = {}
    if (
        singleton_round33_diagnostic is not None
        and singleton_round33_plot_pdf is not None
        and singleton_round33_png is not None
    ):
        baseline_ids = set(MATCHED_SINGLETON_EXECUTION_IDS)
        matched_baseline_sources = [
            dict(_mapping(raw, label="singleton round-33 baseline source"))
            for raw in _sequence(
                source_provenance.get("included_sources"),
                label="cross-revision included sources",
            )
            if _mapping(
                raw,
                label="singleton round-33 candidate baseline source",
            ).get("execution_id")
            in baseline_ids
        ]
        if {
            str(row.get("execution_id"))
            for row in matched_baseline_sources
        } != baseline_ids:
            raise ReportInputError(
                "singleton round-33 baseline source provenance drifted"
            )
        singleton_round33_provenance = {
            **singleton_round33_diagnostic,
            "matched_baseline_sources": matched_baseline_sources,
            "compile_path_policy": (
                "all_three_authenticated_round33_prefixes_recompiled_by_"
                "common_paper_i_qiskit_compiler_v1"
            ),
        }
        limitations.extend(
            (
                (
                    "The strong-strong singleton round-33 page admits no "
                    "additional 48-cell result. Its current-route curve is "
                    "an authenticated partial local checkpoint."
                ),
                (
                    "The round-33 singleton comparison is whole-route, not "
                    "a one-variable cost ablation: global candidate supply "
                    "and all-phase Qiskit ranking differ from staged "
                    "macro-to-singleton proxy ranking."
                ),
                (
                    "All three round-33 tuples use the same Paper-I Qiskit "
                    "prefix compilation path; the current run remains "
                    "incomplete at 33 of 50 requested rounds."
                ),
            )
        )
        singleton_round33_outputs = {
            "strong_strong_singleton_round33_plot_pdf": {
                "path": str(singleton_round33_plot_pdf),
                "sha256": _sha256_file(singleton_round33_plot_pdf),
            },
            "strong_strong_singleton_round33_plot_png": {
                "path": str(singleton_round33_png),
                "sha256": _sha256_file(singleton_round33_png),
            },
        }
    provenance = {
        "schema": (
            "paper_i_ra_adapt_stationary_core_master_"
            "cross_revision_partial_progress_v1"
        ),
        "report_identity": CROSS_REVISION_STEM,
        "package_ids": package_ids,
        "status": (
            "partial_cross_revision_observed_results_with_explicit_"
            "recovery_or_paused_prefix_non_evidentiary"
            if recovery_count or local_paused_count
            else "partial_cross_revision_validated_results_non_evidentiary"
        ),
        "partial_progress": True,
        "cross_revision": True,
        "not_paper_evidence": True,
        "paper_evidence_adopted": False,
        "canonical_results_pdf_emitted": False,
        "final_selection_consumed": False,
        "included_count": len(included_execution_ids),
        "ordinary_validated_count": ordinary_validated_count,
        "recovered_count": recovery_count,
        "recovery_counts": recovery_counts,
        "local_paused_prefix_count": local_paused_count,
        "local_paused_prefix_sources": local_paused_sources,
        "pending_count": len(missing_execution_ids),
        "included_execution_ids": included_execution_ids,
        "missing_execution_ids": missing_execution_ids,
        "metric": "same_cutoff_absolute_energy_error",
        "display_rounds": list(range(0, 51)),
        "parameter_manifest": parameter_manifest,
        "terminal_cost_policy": _terminal_cost_policy(),
        "layout": layout,
        **source_provenance,
        "diagnostic_comparison": diagnostic_provenance,
        "global_singleton_weak_weak_comparison": (
            global_singleton_provenance
        ),
        "strong_strong_singleton_round33_comparison": (
            singleton_round33_provenance
        ),
        "intermediate_strong_macro_cumulative_plateau_comparison": (
            cumulative_plateau_macro_diagnostic
        ),
        "limitations": limitations,
        "outputs": {
            "partial_progress_pdf": {
                "path": str(pdf),
                "sha256": _sha256_file(pdf),
            },
            "macro_png": {
                "path": str(macro_png),
                "sha256": _sha256_file(macro_png),
            },
            "macro_plot_pdf": {
                "path": str(macro_plot_pdf),
                "sha256": _sha256_file(macro_plot_pdf),
            },
            "singleton_png": {
                "path": str(singleton_png),
                "sha256": _sha256_file(singleton_png),
            },
            "singleton_plot_pdf": {
                "path": str(singleton_plot_pdf),
                "sha256": _sha256_file(singleton_plot_pdf),
            },
            **diagnostic_outputs,
            **global_singleton_outputs,
            **singleton_round33_outputs,
            "tex": {"path": str(tex), "sha256": _sha256_file(tex)},
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pdf, provenance_path


def build_final_report(
    *,
    selection_path: Path,
    validation_path: Path,
    fetched_dir: Path,
    output_dir: Path | None = None,
    package_dir: Path | None = None,
    terminal_qiskit_compiler: Callable[[Any], Any] | None = None,
) -> tuple[Path, Path]:
    _configure_package_dir(PACKAGE_DIR if package_dir is None else package_dir)
    if output_dir is None:
        output_dir = OUTPUT_DIR
    cells, selection_sources = load_selected_cells(
        selection_path=selection_path,
        validation_path=validation_path,
        fetched_dir=fetched_dir,
        terminal_qiskit_compiler=terminal_qiskit_compiler,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_manifest = _parameter_manifest()
    macro_png = output_dir / f"{STEM}_macro_master.png"
    singleton_png = output_dir / f"{STEM}_singleton_master.png"
    macro_plot_pdf = output_dir / f"{STEM}_macro_plots.pdf"
    singleton_plot_pdf = output_dir / f"{STEM}_singleton_plots.pdf"
    _render_plot_grid(
        cells=cells,
        representation_key="macro",
        vector_output=macro_plot_pdf,
        png_output=macro_png,
        pending=False,
    )
    _render_plot_grid(
        cells=cells,
        representation_key="singleton",
        vector_output=singleton_plot_pdf,
        png_output=singleton_png,
        pending=False,
    )
    tex = _write_tex(
        output_dir=output_dir,
        document_stem=STEM,
        macro_plot_pdf=macro_plot_pdf,
        singleton_plot_pdf=singleton_plot_pdf,
        cells=cells,
        parameter_manifest=parameter_manifest,
        pending=False,
    )
    pdf = _compile_tex(tex)
    provenance_path = output_dir / f"{STEM}_provenance.json"
    provenance = {
        "schema": "paper_i_ra_adapt_stationary_core_master_report_v1",
        "package_id": PACKAGE_ID,
        "status": "complete_selected_results_report",
        "paper_evidence_adopted": False,
        "metric": "same_cutoff_absolute_energy_error",
        "display_rounds": list(range(0, 51)),
        "marker_policy": (
            "first_effective_plateau_prefix_when_serialized_"
            "otherwise_terminal_observed_point"
        ),
        "parameter_manifest": parameter_manifest,
        "curve_styles": {
            method: {
                "label": METHODS[method]["label"],
                "color": METHODS[method]["color"],
                "linestyle": "solid",
                "marker": METHODS[method]["marker"],
            }
            for method in METHOD_ORDER
        },
        "terminal_cost_policy": _terminal_cost_policy(),
        "layout": {
            "page_count": 2,
            "page_1": "macro_generator_v1",
            "page_2": "single_pauli_word_v1",
            "regime_count_per_page": 6,
            "route_count_per_regime": 4,
            "terminal_rows_per_page": 24,
        },
        "package_sources": _package_sources(),
        **selection_sources,
        "limitations": [
            (
                "The v10 source authority preserves the declared stationary "
                "active-gradient policy/accounting projection; the report "
                "does not infer unrecorded per-round evidence."
            ),
            (
                "Report construction selects and displays validated attempts "
                "but does not adopt them as paper evidence."
            ),
        ],
        "outputs": {
            "pdf": {"path": str(pdf), "sha256": _sha256_file(pdf)},
            "macro_png": {
                "path": str(macro_png),
                "sha256": _sha256_file(macro_png),
            },
            "macro_plot_pdf": {
                "path": str(macro_plot_pdf),
                "sha256": _sha256_file(macro_plot_pdf),
            },
            "singleton_png": {
                "path": str(singleton_png),
                "sha256": _sha256_file(singleton_png),
            },
            "singleton_plot_pdf": {
                "path": str(singleton_plot_pdf),
                "sha256": _sha256_file(singleton_plot_pdf),
            },
            "tex": {"path": str(tex), "sha256": _sha256_file(tex)},
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pdf, provenance_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preview", action="store_true")
    modes.add_argument("--partial-progress", action="store_true")
    modes.add_argument("--cross-revision-progress", action="store_true")
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=DEFAULT_PACKAGE_DIR,
        help=(
            "sealed stationary-core package; report identity and "
            "archive paths are derived from its manifest"
        ),
    )
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--fetched-dir", type=Path)
    parser.add_argument(
        "--partial-source",
        action="append",
        nargs=4,
        metavar=(
            "METHOD_FAMILY",
            "PACKAGE_DIR",
            "VALIDATION",
            "FETCHED_DIR",
        ),
        help=(
            "repeatable cross-revision source: append|ra, sealed package, "
            "validated receipt, and fetched archive directory"
        ),
    )
    parser.add_argument(
        "--recovery-adapter",
        action="append",
        type=Path,
        help=(
            "repeatable explicit non-evidentiary recovery adapter; "
            "cross-revision progress only"
        ),
    )
    parser.add_argument(
        "--diagnostic-qiskit-plateau-run-dir",
        type=Path,
        help=(
            "optional completed local strong-weak macro Qiskit-cost "
            "plateau run; cross-revision progress only"
        ),
    )
    parser.add_argument(
        "--diagnostic-qiskit-plateau-log",
        type=Path,
        help=(
            "remote-runner log paired with "
            "--diagnostic-qiskit-plateau-run-dir"
        ),
    )
    parser.add_argument(
        "--diagnostic-qiskit-always-run-dir",
        type=Path,
        help=(
            "completed local strong-weak macro commutation-reduced "
            "always-insertion Qiskit-cost run; cross-revision progress only"
        ),
    )
    parser.add_argument(
        "--global-singleton-weak-weak-adapter",
        type=Path,
        help=(
            "authenticated diagnostic adapter for the completed weak-weak "
            "global-singleton append-versus-plateau pair; "
            "cross-revision progress only"
        ),
    )
    parser.add_argument(
        "--diagnostic-qiskit-singleton-run-dir",
        type=Path,
        help=(
            "preserved local strong-strong global-singleton Qiskit-cost "
            "checkpoint for the matched round-33 page; cross-revision "
            "progress only"
        ),
    )
    parser.add_argument(
        "--diagnostic-local-always-prefix",
        action="append",
        nargs=3,
        metavar=("JOB", "CHECKPOINT", "LOG"),
        help=(
            "repeatable source-locked local RA-always prefix deliberately "
            "stopped before round 50; cross-revision progress only"
        ),
    )
    parser.add_argument(
        "--diagnostic-cumulative-plateau-macro-run-dir",
        type=Path,
        help=(
            "completed local intermediate-strong macro cumulative-relative "
            "plateau diagnostic; cross-revision progress only"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        package_dir = args.package_dir.resolve()
        _configure_package_dir(package_dir)
        output_dir = (
            (
                CROSS_REVISION_OUTPUT_DIR
                if args.cross_revision_progress
                else OUTPUT_DIR
            )
            if args.output_dir is None
            else args.output_dir.resolve()
        )
        if args.preview:
            if any(
                value is not None
                for value in (
                    args.selection,
                    args.validation,
                    args.fetched_dir,
                    args.partial_source,
                    args.recovery_adapter,
                    args.diagnostic_qiskit_plateau_run_dir,
                    args.diagnostic_qiskit_plateau_log,
                    args.diagnostic_qiskit_always_run_dir,
                    args.global_singleton_weak_weak_adapter,
                    args.diagnostic_qiskit_singleton_run_dir,
                    args.diagnostic_local_always_prefix,
                    args.diagnostic_cumulative_plateau_macro_run_dir,
                )
            ):
                raise ReportInputError(
                    "preview mode does not consume runtime result paths"
                )
            pdf, sidecar = build_pending_preview(
                output_dir=output_dir,
                package_dir=package_dir,
            )
        elif args.cross_revision_progress:
            if (
                args.selection is not None
                or args.validation is not None
                or args.fetched_dir is not None
                or not args.partial_source
            ):
                raise ReportInputError(
                    "cross-revision progress requires repeated "
                    "--partial-source entries and does not consume "
                    "--selection/--validation/--fetched-dir"
                )
            source_specs = [
                {
                    "method_family": row[0],
                    "package_dir": Path(row[1]).resolve(),
                    "validation_path": Path(row[2]).resolve(),
                    "fetched_dir": Path(row[3]).resolve(),
                }
                for row in args.partial_source
            ]
            pdf, sidecar = build_cross_revision_partial_progress(
                source_specs=source_specs,
                recovery_adapter_paths=tuple(
                    value.resolve()
                    for value in (args.recovery_adapter or ())
                ),
                diagnostic_local_always_prefixes=tuple(
                    {
                        "job_path": Path(row[0]).resolve(),
                        "checkpoint_path": Path(row[1]).resolve(),
                        "log_path": Path(row[2]).resolve(),
                    }
                    for row in (
                        args.diagnostic_local_always_prefix or ()
                    )
                ),
                output_dir=output_dir,
                diagnostic_qiskit_plateau_run_dir=(
                    None
                    if args.diagnostic_qiskit_plateau_run_dir is None
                    else args.diagnostic_qiskit_plateau_run_dir.resolve()
                ),
                diagnostic_qiskit_plateau_log=(
                    None
                    if args.diagnostic_qiskit_plateau_log is None
                    else args.diagnostic_qiskit_plateau_log.resolve()
                ),
                diagnostic_qiskit_always_run_dir=(
                    None
                    if args.diagnostic_qiskit_always_run_dir is None
                    else args.diagnostic_qiskit_always_run_dir.resolve()
                ),
                global_singleton_weak_weak_adapter_path=(
                    None
                    if args.global_singleton_weak_weak_adapter is None
                    else args.global_singleton_weak_weak_adapter.resolve()
                ),
                diagnostic_qiskit_singleton_run_dir=(
                    None
                    if args.diagnostic_qiskit_singleton_run_dir is None
                    else args.diagnostic_qiskit_singleton_run_dir.resolve()
                ),
                diagnostic_cumulative_plateau_macro_run_dir=(
                    None
                    if args.diagnostic_cumulative_plateau_macro_run_dir
                    is None
                    else args.diagnostic_cumulative_plateau_macro_run_dir.resolve()
                ),
            )
        elif args.partial_progress:
            if (
                args.selection is not None
                or args.validation is None
                or args.fetched_dir is None
                or args.partial_source is not None
                or args.recovery_adapter is not None
                or args.diagnostic_qiskit_plateau_run_dir is not None
                or args.diagnostic_qiskit_plateau_log is not None
                or args.diagnostic_qiskit_always_run_dir is not None
                or args.global_singleton_weak_weak_adapter is not None
                or args.diagnostic_qiskit_singleton_run_dir is not None
                or args.diagnostic_local_always_prefix is not None
                or args.diagnostic_cumulative_plateau_macro_run_dir is not None
            ):
                raise ReportInputError(
                    "partial-progress mode requires --validation and "
                    "--fetched-dir and does not consume --selection"
                )
            pdf, sidecar = build_partial_progress(
                validation_path=args.validation.resolve(),
                fetched_dir=args.fetched_dir.resolve(),
                output_dir=output_dir,
                package_dir=package_dir,
            )
        else:
            if (
                args.selection is None
                or args.validation is None
                or args.fetched_dir is None
                or args.partial_source is not None
                or args.recovery_adapter is not None
                or args.diagnostic_qiskit_plateau_run_dir is not None
                or args.diagnostic_qiskit_plateau_log is not None
                or args.diagnostic_qiskit_always_run_dir is not None
                or args.global_singleton_weak_weak_adapter is not None
                or args.diagnostic_qiskit_singleton_run_dir is not None
                or args.diagnostic_local_always_prefix is not None
                or args.diagnostic_cumulative_plateau_macro_run_dir is not None
            ):
                raise ReportInputError(
                    "final mode requires --selection, --validation, and "
                    "--fetched-dir"
                )
            pdf, sidecar = build_final_report(
                selection_path=args.selection.resolve(),
                validation_path=args.validation.resolve(),
                fetched_dir=args.fetched_dir.resolve(),
                output_dir=output_dir,
                package_dir=package_dir,
            )
        print(pdf)
        print(sidecar)
        return 0
    except (OSError, ReportInputError, RuntimeError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

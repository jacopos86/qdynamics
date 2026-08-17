#!/usr/bin/env python3
"""Replace page 6 with the recovered RA/Append singleton R70 comparison.

Append-ADAPT inputs are complete authenticated 0..70 adapters.  The RA jobs
completed 70 controller transitions, but their final artifact publication
failed after an EXDEV directory rename.  Their scheduler stdout therefore
recovers accepted energies only for rounds 0..69: the event emitted at depth
``d`` records ``energy_before_refit``, which is the accepted energy entering
round ``d``.  This distinction is enforced here and is visible on the page.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tarfile
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import (
    add_paper_i_append_r70_singleton_progress_page as legacy_page,
)


COMBINED_ADAPTER_SCHEMA = "paper_i_ra_append_singleton_r70_page6_adapter_v1"
APPEND_ADAPTER_SCHEMA = "paper_i_append_adapt_singleton_r70_progress_adapter_v1"
COST_ADAPTER_SCHEMA = "paper_i_ra_append_singleton_r70_cost_diagnostic_v1"
PRIOR_PAGE_IDS = (
    "ra_historical_average_vs_append_singleton_r70_progress_v1",
    "ra_historical_average_vs_append_singleton_r70_costs_v2",
)
PAGE_ID = "ra_historical_average_vs_append_singleton_r70_costs_v3"
RA_PACKAGE_ID = (
    "paper_i_ra_adapt_historical_average_singleton_plateau6_"
    "r70_fresh_20260801_v4_chtc"
)
RA_CLUSTER_ID = 9_400_249
REGIME_ORDER = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
REGIME_LABELS = {
    "weak_weak": "Weak--weak",
    "intermediate_weak": "Intermediate--weak",
    "strong_weak_u8": "Strong--weak",
    "weak_strong": "Weak--strong",
    "intermediate_strong": "Intermediate--strong",
    "strong_strong_u8": "Strong--strong",
}
REGIME_ABBREVIATIONS = {
    "weak_weak": "WW",
    "intermediate_weak": "IW",
    "strong_weak_u8": "SW",
    "weak_strong": "WS",
    "intermediate_strong": "IS",
    "strong_strong_u8": "SS",
}
NPH_BY_REGIME = {
    "weak_weak": 3,
    "intermediate_weak": 3,
    "strong_weak_u8": 3,
    "weak_strong": 7,
    "intermediate_strong": 7,
    "strong_strong_u8": 7,
}
PROC_BY_REGIME = {regime: index for index, regime in enumerate(REGIME_ORDER)}
QISKIT_FIELDS = ("N2q", "D2q", "Dc", "W1q")
COST_FIELDS = (*QISKIT_FIELDS, "S_alg")
OLD_PAGE_LIMITATION = (
    "Page 6 is a supplemental fresh Append-ADAPT singleton R70 progress "
    "diagnostic; it does not enter the validated 48-cell stationary-core "
    "matrix and is not adopted Paper-I evidence."
)
NEW_PAGE_LIMITATION = (
    "Page 6 compares complete authenticated Append-ADAPT singleton rounds "
    "0--70 with scheduler-stdout-recovered historical-average stationary "
    "RA plateau energies for rounds 0--69. RA round 70 was accepted, but its "
    "post-refit energy, ledger, checkpoint, and result were lost in the "
    "post-run EXDEV publication failure; the page is diagnostic and is not "
    "adopted Paper-I evidence."
)
RA_S_ALG_LIMITATION = (
    "RA S_alg is unavailable: the EXDEV publication failure prevented "
    "checkpoint and estimator-ledger transfer; retained stdout has energies "
    "and selected operators only, not executed scalar-estimator occurrences. "
    "No estimate is substituted."
)
ATTEMPT_MEMBERS = {
    "worker_outputs/attempt_identity.tsv",
    "worker_outputs/worker_exit_status.txt",
    "authority/job.json",
    "authority/execution_authorization.json",
    "authority/activation_manifest.json",
    "worker_attempt_receipt.json",
}
ASSET_STEM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
RA_ATTEMPT_SCHEMA = (
    "paper_i_ra_adapt_historical_average_singleton_plateau6_r70_"
    "worker_attempt_v1"
)


class Page6InputError(ValueError):
    """Raised when recovered sources cannot support the diagnostic page."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "sha256" in result:
        raise Page6InputError("self-digest input already contains sha256")
    result["sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    observed = value.get("sha256")
    unsigned = copy.deepcopy(dict(value))
    unsigned.pop("sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if observed != expected:
        raise Page6InputError(f"{label} self-digest drifted")
    return str(observed)


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


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Page6InputError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise Page6InputError(f"{label} must be a JSON object")
    return value


def mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Page6InputError(f"{label} must be an object")
    return value


def sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise Page6InputError(f"{label} must be an array")
    return value


def safe_relative_path(value: str, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise Page6InputError(f"{label} must be a normalized relative path")
    return path


def integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Page6InputError(f"{label} must be an integer >= {minimum}")
    return value


def finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Page6InputError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Page6InputError(f"{label} must be finite")
    return result


def execution_id(regime: str) -> str:
    return (
        f"historical_average_v4_r70_fresh__{regime}__"
        f"nph{NPH_BY_REGIME[regime]}__ra_singleton_plateau"
    )


def validate_attempt_archive(path: Path, *, regime: str) -> dict[str, Any]:
    expected_id = execution_id(regime)
    expected_name = (
        f"{expected_id}__cluster_{RA_CLUSTER_ID}__proc_"
        f"{PROC_BY_REGIME[regime]}.tar.gz"
    )
    if path.name != expected_name or not path.is_file() or path.is_symlink():
        raise Page6InputError(f"{regime}: attempt archive identity drifted")
    payloads: dict[str, bytes] = {}
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            names: set[str] = set()
            for member in archive:
                name = member.name
                if (
                    name in names
                    or name.startswith("/")
                    or ".." in Path(name).parts
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                ):
                    raise Page6InputError(f"{regime}: unsafe attempt member")
                names.add(name)
                stream = archive.extractfile(member)
                if stream is None:
                    raise Page6InputError(f"{regime}: unreadable attempt member")
                payloads[name] = stream.read()
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise Page6InputError(f"{regime}: attempt archive is invalid") from exc
    if set(payloads) != ATTEMPT_MEMBERS:
        raise Page6InputError(f"{regime}: attempt member closure drifted")
    status_text = payloads["worker_outputs/worker_exit_status.txt"].decode(
        "ascii"
    ).strip()
    if status_text != "2":
        raise Page6InputError(f"{regime}: expected post-run exit status 2")
    try:
        receipt = json.loads(payloads["worker_attempt_receipt.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Page6InputError(f"{regime}: attempt receipt is invalid") from exc
    if not isinstance(receipt, dict):
        raise Page6InputError(f"{regime}: attempt receipt is invalid")
    canonical = verify_self_digest(receipt, label=f"{regime} attempt receipt")
    if (
        receipt.get("schema") != RA_ATTEMPT_SCHEMA
        or receipt.get("execution_id") != expected_id
        or receipt.get("cluster_id") != RA_CLUSTER_ID
        or receipt.get("proc_id") != PROC_BY_REGIME[regime]
        or receipt.get("worker_exit_status") != 2
    ):
        raise Page6InputError(f"{regime}: attempt receipt identity drifted")
    attempt_ordinal = integer(
        receipt.get("attempt_ordinal"),
        label=f"{regime} attempt ordinal",
        minimum=1,
    )
    expected_identity = (
        f"{expected_id}\t{RA_CLUSTER_ID}\t{PROC_BY_REGIME[regime]}\t"
        f"{attempt_ordinal}\n"
    ).encode("ascii")
    if payloads["worker_outputs/attempt_identity.tsv"] != expected_identity:
        raise Page6InputError(f"{regime}: attempt identity marker drifted")

    worker_rows = sequence(
        receipt.get("worker_files"), label=f"{regime} worker file bindings"
    )
    expected_worker_payloads = {
        name.removeprefix("worker_outputs/"): payload
        for name, payload in payloads.items()
        if name.startswith("worker_outputs/")
    }
    observed_worker_paths: set[str] = set()
    for index, raw_row in enumerate(worker_rows):
        row = mapping(raw_row, label=f"{regime} worker binding {index}")
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or relative in observed_worker_paths
            or relative not in expected_worker_payloads
        ):
            raise Page6InputError(f"{regime}: worker binding closure drifted")
        observed_worker_paths.add(relative)
        payload = expected_worker_payloads[relative]
        if (
            row.get("sha256") != hashlib.sha256(payload).hexdigest()
            or row.get("size_bytes") != len(payload)
        ):
            raise Page6InputError(f"{regime}: worker binding bytes drifted")
    if observed_worker_paths != set(expected_worker_payloads):
        raise Page6InputError(f"{regime}: worker binding closure drifted")

    authority_fields = {
        "job_file_sha256": "authority/job.json",
        "authorization_file_sha256": "authority/execution_authorization.json",
        "activation_manifest_file_sha256": "authority/activation_manifest.json",
    }
    for receipt_field, member_name in authority_fields.items():
        if receipt.get(receipt_field) != hashlib.sha256(
            payloads[member_name]
        ).hexdigest():
            raise Page6InputError(f"{regime}: authority binding bytes drifted")
    return {
        **file_binding(path),
        "attempt_receipt_canonical_sha256": canonical,
        "attempt_ordinal": attempt_ordinal,
    }


def parse_ra_stdout(
    path: Path, *, regime: str, exact_energy: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise Page6InputError(f"{regime}: stdout is unavailable")
    events: dict[int, Mapping[str, Any]] = {}
    attempt_packaging_result: Mapping[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if raw_line.startswith("AI_LOG "):
            event = json.loads(raw_line.removeprefix("AI_LOG "))
            if (
                not isinstance(event, dict)
                or event.get("event") != "hardcoded_adapt_iter"
            ):
                continue
            depth = integer(event.get("depth"), label=f"{regime} depth", minimum=1)
            if depth in events:
                raise Page6InputError(f"{regime}: stdout duplicates depth {depth}")
            events[depth] = event
        elif raw_line.startswith("{") and '"output_archive"' in raw_line:
            candidate = json.loads(raw_line)
            if isinstance(candidate, dict):
                attempt_packaging_result = candidate
    if tuple(sorted(events)) != tuple(range(1, 71)):
        raise Page6InputError(f"{regime}: stdout depths are not exactly 1..70")
    if (
        attempt_packaging_result is None
        or attempt_packaging_result.get("status") != "passed"
    ):
        raise Page6InputError(f"{regime}: attempt packaging completion is unavailable")

    points: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    prior_energy: float | None = None
    for depth in range(1, 71):
        event = events[depth]
        energy = finite(event.get("energy"), label=f"{regime} depth {depth} energy")
        if prior_energy is not None and energy > prior_energy + 1.0e-10:
            raise Page6InputError(f"{regime}: recovered accepted energy increased")
        prior_energy = energy
        # Direct-controller hardcoded_adapt_iter is emitted after accepting
        # round depth, but its energy field is artifacts.energy_before_refit.
        # It therefore binds the accepted energy at round depth - 1.
        points.append(
            {
                "round": depth - 1,
                "energy": energy,
                "delta_e": abs(energy - exact_energy),
            }
        )
        label = event.get("best_op")
        if not isinstance(label, str) or not label:
            raise Page6InputError(f"{regime}: selected generator is unavailable")
        decisions.append(
            {
                "accepted_round": depth,
                "candidate_label": label,
                "selected_position": integer(
                    event.get("selected_position"),
                    label=f"{regime} depth {depth} insertion position",
                ),
            }
        )
    return points, decisions, {
        "stdout": file_binding(path),
        "attempt_packaging_result": copy.deepcopy(
            dict(attempt_packaging_result)
        ),
    }


def validate_append_adapter(path: Path) -> dict[str, Any]:
    adapter = load_json(path, label="Append R70 adapter")
    canonical = verify_self_digest(adapter, label="Append R70 adapter")
    if (
        adapter.get("schema") != APPEND_ADAPTER_SCHEMA
        or adapter.get("status") != "passed"
        or tuple(adapter.get("regime_order", ())) != REGIME_ORDER
        or tuple(adapter.get("completed_regimes", ())) != REGIME_ORDER
        or tuple(adapter.get("pending_regimes", ())) != ()
    ):
        raise Page6InputError("Append adapter is not the complete six-regime R70 set")
    cells = sequence(adapter.get("cells"), label="Append cells")
    by_regime = {
        str(mapping(cell, label="Append cell").get("regime_id")): cell
        for cell in cells
    }
    if set(by_regime) != set(REGIME_ORDER) or len(cells) != 6:
        raise Page6InputError("Append adapter regime closure drifted")
    for regime in REGIME_ORDER:
        cell = mapping(by_regime[regime], label=f"Append {regime}")
        points = sequence(cell.get("points"), label=f"Append {regime} points")
        point_rounds = [
            mapping(point, label=f"Append {regime} point {index}").get("round")
            for index, point in enumerate(points)
        ]
        if len(points) != 71 or point_rounds != list(range(71)):
            raise Page6InputError(f"Append {regime} points are not rounds 0..70")
        endpoints = mapping(cell.get("endpoints"), label=f"Append {regime} endpoints")
        for endpoint_round in (50, 70):
            endpoint = mapping(
                endpoints.get(f"round_{endpoint_round}"),
                label=f"Append {regime} round {endpoint_round}",
            )
            costs = mapping(endpoint.get("costs"), label="Append costs")
            if set(costs) != set(COST_FIELDS):
                raise Page6InputError(f"Append {regime} cost tuple drifted")
    return {
        **copy.deepcopy(adapter),
        "sha256": canonical,
        "file_binding": file_binding(path),
    }


def build_combined_adapter(
    *,
    append_adapter_path: Path,
    ra_retrieval_root: Path,
    ra_package_dir: Path,
    output: Path,
) -> dict[str, Any]:
    append = validate_append_adapter(append_adapter_path)
    package_manifest_path = ra_package_dir / "package_manifest.json"
    package_manifest = load_json(package_manifest_path, label="RA package manifest")
    package_canonical = verify_self_digest(
        package_manifest, label="RA package manifest"
    )
    if package_manifest.get("package_id") != RA_PACKAGE_ID:
        raise Page6InputError("RA package identity drifted")
    append_cells = {str(cell["regime_id"]): cell for cell in append["cells"]}
    cells: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        proc = PROC_BY_REGIME[regime]
        execution = execution_id(regime)
        job_path = ra_package_dir / "jobs" / f"{execution}.json"
        job = load_json(job_path, label=f"{regime} RA job")
        job_canonical = verify_self_digest(job, label=f"{regime} RA job")
        if (
            job.get("execution_id") != execution
            or job.get("regime_id") != regime
            or job.get("target_horizon") != 70
            or job.get("exact_same_cutoff_energy") is None
        ):
            raise Page6InputError(f"{regime}: RA job contract drifted")
        exact_energy = finite(
            job["exact_same_cutoff_energy"], label=f"{regime} exact energy"
        )
        append_cell = copy.deepcopy(dict(append_cells[regime]))
        if not math.isclose(
            exact_energy,
            finite(
                append_cell.get("exact_same_cutoff_energy"),
                label="Append exact energy",
            ),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise Page6InputError(f"{regime}: RA/Append exact reference drifted")
        log_stem = f"{RA_CLUSTER_ID}.{proc}__{execution}"
        stdout_path = ra_retrieval_root / "logs" / f"{log_stem}.out"
        stderr_path = ra_retrieval_root / "logs" / f"{log_stem}.err"
        condor_log_path = ra_retrieval_root / "logs" / f"{log_stem}.log"
        if not stderr_path.is_file() or not condor_log_path.is_file():
            raise Page6InputError(f"{regime}: scheduler logs are incomplete")
        stderr_text = stderr_path.read_text(encoding="utf-8")
        if (
            "Invalid cross-device link" not in stderr_text
            or "worker_outputs/artifacts" not in stderr_text
        ):
            raise Page6InputError(f"{regime}: expected EXDEV failure is absent")
        points, decisions, stdout_source = parse_ra_stdout(
            stdout_path, regime=regime, exact_energy=exact_energy
        )
        attempt_path = (
            ra_retrieval_root
            / "fetched"
            / f"{execution}__cluster_{RA_CLUSTER_ID}__proc_{proc}.tar.gz"
        )
        attempt = validate_attempt_archive(attempt_path, regime=regime)
        packaging_result = mapping(
            stdout_source["attempt_packaging_result"],
            label=f"{regime} attempt packaging result",
        )
        expected_archive_relative = (
            f"transfer/{execution}__{RA_CLUSTER_ID}__{proc}.tar.gz"
        )
        if (
            packaging_result.get("output_archive") != expected_archive_relative
            or packaging_result.get("output_archive_sha256") != attempt["sha256"]
            or packaging_result.get("output_archive_size_bytes")
            != attempt["size_bytes"]
            or packaging_result.get("worker_attempt_receipt_sha256")
            != attempt["attempt_receipt_canonical_sha256"]
        ):
            raise Page6InputError(
                f"{regime}: stdout/archive attempt binding drifted"
            )
        cells.append(
            {
                "regime_id": regime,
                "display_name": REGIME_LABELS[regime],
                "nph": NPH_BY_REGIME[regime],
                "exact_same_cutoff_energy": exact_energy,
                "append": append_cell,
                "ra_historical_average_plateau": {
                    "execution_id": execution,
                    "source_classification": "diagnostic_from_scheduler_stdout_v1",
                    "accepted_controller_rounds_observed": 70,
                    "recoverable_energy_rounds": {"minimum": 0, "maximum": 69},
                    "round_70_energy_available": False,
                    "round_70_cost_tuple_available": False,
                    "points": points,
                    "accepted_decisions": decisions,
                    "source": {
                        **stdout_source,
                        "stderr": file_binding(stderr_path),
                        "condor_log": file_binding(condor_log_path),
                        "attempt_archive": attempt,
                        "job": {
                            **file_binding(job_path),
                            "canonical_sha256": job_canonical,
                        },
                    },
                },
            }
        )
    combined = digested(
        {
            "schema": COMBINED_ADAPTER_SCHEMA,
            "status": "passed_with_explicit_ra_terminal_limitation",
            "classification": "diagnostic_not_adopted_evidence",
            "regime_order": list(REGIME_ORDER),
            "append_adapter": {
                **append["file_binding"],
                "canonical_sha256": append["sha256"],
            },
            "ra_package": {
                **file_binding(package_manifest_path),
                "canonical_sha256": package_canonical,
                "package_id": RA_PACKAGE_ID,
                "cluster_id": RA_CLUSTER_ID,
            },
            "same_cutoff_reference": copy.deepcopy(append["same_cutoff_reference"]),
            "display_rounds": {"minimum": 0, "maximum": 70},
            "ra_exact_energy_rounds": {"minimum": 0, "maximum": 69},
            "append_exact_energy_rounds": {"minimum": 0, "maximum": 70},
            "limitations": [NEW_PAGE_LIMITATION],
            "cells": cells,
        }
    )
    if output.exists() or output.is_symlink():
        existing = load_json(output, label="existing combined adapter")
        verify_self_digest(existing, label="existing combined adapter")
        if canonical_json_bytes(existing) != canonical_json_bytes(combined):
            raise Page6InputError("refusing to replace a different combined adapter")
        return existing
    legacy_page._atomic_write_json(output, combined)
    return combined


def format_delta_e(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\!\times\!10^{{{int(exponent)}}}$"


def _point_by_round(points: Sequence[Any], *, label: str) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for index, raw in enumerate(points):
        point = mapping(raw, label=f"{label} point {index}")
        round_index = integer(point.get("round"), label=f"{label} round")
        if round_index in result:
            raise Page6InputError(f"{label} duplicates round {round_index}")
        result[round_index] = point
    return result


def _effective_plateau(points: Sequence[Any], *, label: str) -> int:
    by_round = _point_by_round(points, label=label)
    eligible = [
        (round_index, finite(point.get("delta_e"), label=f"{label} delta E"))
        for round_index, point in sorted(by_round.items())
        if round_index >= 1
    ]
    if not eligible:
        raise Page6InputError(f"{label} has no positive-round trajectory")
    threshold = 1.10 * min(error for _round, error in eligible)
    return next(round_index for round_index, error in eligible if error <= threshold)


def _validate_cost_observation(
    value: Any, *, label: str, s_alg_required: bool
) -> Mapping[str, Any]:
    observation = mapping(value, label=label)
    costs = mapping(observation.get("costs"), label=f"{label} costs")
    if set(costs) != set(COST_FIELDS):
        raise Page6InputError(f"{label} cost tuple drifted")
    for field in QISKIT_FIELDS:
        integer(costs.get(field), label=f"{label} {field}")
    s_alg = costs.get("S_alg")
    if s_alg_required:
        s_alg_value = integer(s_alg, label=f"{label} S_alg")
        components = mapping(
            observation.get("S_alg_components"),
            label=f"{label} S_alg components",
        )
        component_total = sum(
            integer(components.get(field), label=f"{label} {field}")
            for field in ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
        )
        if component_total != s_alg_value:
            raise Page6InputError(f"{label} S_alg components do not close")
    elif (
        s_alg is not None
        or observation.get("S_alg_status")
        != "unavailable_exdev_estimator_ledger_not_transferred"
    ):
        raise Page6InputError(f"{label} must preserve the RA S_alg limitation")
    return observation


def validate_cost_adapter(
    path: Path,
    *,
    combined: Mapping[str, Any],
    combined_path: Path,
    combined_canonical_sha256: str,
) -> dict[str, Any]:
    cost_adapter = load_json(path, label="page-6 cost adapter")
    canonical = verify_self_digest(cost_adapter, label="page-6 cost adapter")
    trajectory_binding = mapping(
        cost_adapter.get("trajectory_adapter"), label="trajectory adapter binding"
    )
    compiler = mapping(cost_adapter.get("compiler"), label="compiler receipt")
    policy = mapping(cost_adapter.get("matching_policy"), label="matching policy")
    sources = mapping(cost_adapter.get("sources"), label="cost sources")
    if (
        cost_adapter.get("schema") != COST_ADAPTER_SCHEMA
        or cost_adapter.get("status") != "passed_with_ra_s_alg_unavailable"
        or tuple(cost_adapter.get("regime_order", ())) != REGIME_ORDER
        or trajectory_binding.get("canonical_sha256")
        != combined_canonical_sha256
        or trajectory_binding.get("sha256") != sha256_file(combined_path)
        or trajectory_binding.get("size_bytes") != combined_path.stat().st_size
        or compiler.get("compile_convention")
        != "table_i_basis_gate_transpile_v1"
        or compiler.get("qiskit_version") != "2.3.1"
        or compiler.get("optimization_level") != 0
        or compiler.get("seed_transpiler") != 7
        or compiler.get("reference_state_included") is not True
        or tuple(compiler.get("basis_gates", ()))
        != ("id", "x", "sx", "rx", "ry", "rz", "h", "s", "sdg", "cx", "cz")
        or policy.get("policy_id")
        != "earlier_effective_plateau_common_accuracy_v1"
        or policy.get("plateau_relative_tolerance") != 0.10
        or policy.get("round_zero_excluded") is not True
        or policy.get("earliest_crossing") is not True
        or RA_S_ALG_LIMITATION not in cost_adapter.get("limitations", ())
    ):
        raise Page6InputError("page-6 cost adapter contract drifted")

    for source_key in ("ra_package", "append_package"):
        source = mapping(sources.get(source_key), label=f"{source_key} source")
        relative = source.get("path")
        if not isinstance(relative, str):
            raise Page6InputError(f"{source_key} path is unavailable")
        manifest_path = REPO_ROOT / safe_relative_path(
            relative, label=f"{source_key} path"
        )
        manifest = load_json(manifest_path, label=f"{source_key} manifest")
        if (
            sha256_file(manifest_path) != source.get("manifest_file_sha256")
            or verify_self_digest(manifest, label=f"{source_key} manifest")
            != source.get("manifest_canonical_sha256")
            or mapping(
                manifest.get("source_archive"), label=f"{source_key} archive"
            ).get("sha256")
            != source.get("source_archive_sha256")
        ):
            raise Page6InputError(f"{source_key} binding drifted")
    reference = mapping(
        sources.get("same_cutoff_reference"), label="same-cutoff reference"
    )
    reference_relative = reference.get("path")
    if not isinstance(reference_relative, str):
        raise Page6InputError("same-cutoff reference path is unavailable")
    reference_path = REPO_ROOT / safe_relative_path(
        reference_relative, label="same-cutoff reference path"
    )
    if sha256_file(reference_path) != reference.get("sha256"):
        raise Page6InputError("same-cutoff reference binding drifted")

    combined_cells = {
        str(mapping(cell, label="combined cell").get("regime_id")): cell
        for cell in sequence(combined.get("cells"), label="combined cells")
    }
    cost_cells = {
        str(mapping(cell, label="cost cell").get("regime_id")): cell
        for cell in sequence(cost_adapter.get("cells"), label="cost cells")
    }
    if set(combined_cells) != set(REGIME_ORDER) or set(cost_cells) != set(REGIME_ORDER):
        raise Page6InputError("page-6 cost adapter regime closure drifted")

    for regime in REGIME_ORDER:
        combined_cell = mapping(combined_cells[regime], label=f"{regime} combined")
        cost_cell = mapping(cost_cells[regime], label=f"{regime} costs")
        ra_points = sequence(
            mapping(
                combined_cell.get("ra_historical_average_plateau"),
                label=f"{regime} RA",
            ).get("points"),
            label=f"{regime} RA points",
        )
        append_points = sequence(
            mapping(combined_cell.get("append"), label=f"{regime} Append").get(
                "points"
            ),
            label=f"{regime} Append points",
        )
        ra_by_round = _point_by_round(ra_points, label=f"{regime} RA")
        append_by_round = _point_by_round(append_points, label=f"{regime} Append")
        endpoint = _validate_cost_observation(
            cost_cell.get("ra_round_69"),
            label=f"{regime} RA round 69",
            s_alg_required=False,
        )
        if (
            endpoint.get("round") != 69
            or not math.isclose(
                finite(endpoint.get("delta_e"), label=f"{regime} RA69 error"),
                finite(ra_by_round[69].get("delta_e"), label=f"{regime} RA69 source"),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise Page6InputError(f"{regime} RA round-69 observation drifted")

        ra_plateau = _effective_plateau(ra_points, label=f"{regime} RA")
        append_plateau = _effective_plateau(append_points, label=f"{regime} Append")
        shared_end = min(ra_plateau, append_plateau)
        ra_min = min(
            finite(point.get("delta_e"), label=f"{regime} RA shared error")
            for round_index, point in ra_by_round.items()
            if 1 <= round_index <= shared_end
        )
        append_min = min(
            finite(point.get("delta_e"), label=f"{regime} Append shared error")
            for round_index, point in append_by_round.items()
            if 1 <= round_index <= shared_end
        )
        target = max(ra_min, append_min)
        ra_crossing = next(
            round_index
            for round_index, point in sorted(ra_by_round.items())
            if 1 <= round_index <= shared_end
            and finite(point.get("delta_e"), label=f"{regime} RA crossing") <= target
        )
        append_crossing = next(
            round_index
            for round_index, point in sorted(append_by_round.items())
            if 1 <= round_index <= shared_end
            and finite(point.get("delta_e"), label=f"{regime} Append crossing")
            <= target
        )
        common = mapping(
            cost_cell.get("common_accuracy"), label=f"{regime} common accuracy"
        )
        ra_observation = _validate_cost_observation(
            common.get("ra"), label=f"{regime} common RA", s_alg_required=False
        )
        append_observation = _validate_cost_observation(
            common.get("append"),
            label=f"{regime} common Append",
            s_alg_required=True,
        )
        expected = (
            ("ra_effective_plateau", ra_plateau),
            ("append_effective_plateau", append_plateau),
            ("shared_window_end", shared_end),
        )
        if any(common.get(field) != value for field, value in expected) or not math.isclose(
            finite(common.get("target_delta_e"), label=f"{regime} target"),
            target,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise Page6InputError(f"{regime} common-accuracy selection drifted")
        for method, observation, crossing, source_points in (
            ("RA", ra_observation, ra_crossing, ra_by_round),
            ("Append", append_observation, append_crossing, append_by_round),
        ):
            if observation.get("round") != crossing or not math.isclose(
                finite(
                    observation.get("delta_e"),
                    label=f"{regime} common {method} error",
                ),
                finite(
                    source_points[crossing].get("delta_e"),
                    label=f"{regime} common {method} source error",
                ),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise Page6InputError(
                    f"{regime} common-accuracy {method} observation drifted"
                )
    return {
        **copy.deepcopy(cost_adapter),
        "sha256": canonical,
        "file_binding": file_binding(path),
    }


def format_costs(value: Mapping[str, Any]) -> str:
    costs = {field: int(value[field]) for field in QISKIT_FIELDS}
    raw_s_alg = value.get("S_alg")
    if raw_s_alg is None:
        s_alg = r"\text{--}"
    else:
        mantissa, exponent = f"{int(raw_s_alg):.1e}".split("e")
        s_alg = rf"{mantissa}\mathrm{{e}}{int(exponent)}"
    display = {
        **costs,
        "S_alg": s_alg,
    }
    return "$({N2q:,},{D2q:,},{Dc:,},{W1q:,},{S_alg})$".format(
        **display,
    )


def latex_escape(value: str) -> str:
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
    return "".join(replacements.get(character, character) for character in value)


def render_plot(adapter: Mapping[str, Any], *, png_path: Path, pdf_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

    cells = {str(cell["regime_id"]): cell for cell in adapter["cells"]}
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.2,
            "axes.labelsize": 8.4,
            "axes.titlesize": 9.3,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.1, 4.1), constrained_layout=True)
    for index, regime in enumerate(REGIME_ORDER):
        ax = axes.flat[index]
        cell = cells[regime]
        append_points = cell["append"]["points"]
        ra_points = cell["ra_historical_average_plateau"]["points"]
        append_rounds = [int(point["round"]) for point in append_points]
        append_errors = [float(point["delta_e"]) for point in append_points]
        ra_rounds = [int(point["round"]) for point in ra_points]
        ra_errors = [float(point["delta_e"]) for point in ra_points]
        ax.plot(append_rounds, append_errors, color="#4C78A8", linewidth=1.6)
        ax.plot(ra_rounds, ra_errors, color="#E45756", linewidth=1.7)
        ax.scatter(
            [append_rounds[-1]],
            [append_errors[-1]],
            marker="o",
            color="#4C78A8",
            s=28,
            zorder=4,
        )
        ax.scatter(
            [ra_rounds[-1]],
            [ra_errors[-1]],
            marker="D",
            color="#E45756",
            s=28,
            zorder=4,
        )
        ax.set_title(REGIME_LABELS[regime])
        ax.set_xlim(0, 70)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", linewidth=0.45, alpha=0.34)
        ax.grid(True, which="minor", linewidth=0.25, alpha=0.14)
        if index // 3 == 1:
            ax.set_xlabel("ADAPT iteration")
        if index % 3 == 0:
            ax.set_ylabel(r"Same-cutoff $\Delta E$")
    fig.suptitle(
        "Singleton R70 comparison: historical-average RA plateau vs Append-ADAPT",
        fontsize=11.6,
        fontweight="bold",
    )
    fig.legend(
        handles=(
            Line2D(
                [0], [0], color="#4C78A8", marker="o", label="Append-ADAPT (k=70)"
            ),
            Line2D(
                [0],
                [0],
                color="#E45756",
                marker="D",
                label="RA historical-average plateau (matched replacement k=69)",
            ),
        ),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=8.2,
        title="Marker denotes terminal observed plotted point",
        title_fontsize=7.5,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_page_tex(
    adapter: Mapping[str, Any],
    *,
    cost_adapter: Mapping[str, Any],
    plot_pdf: Path,
    tex_path: Path,
) -> None:
    cost_cells = {str(cell["regime_id"]): cell for cell in cost_adapter["cells"]}
    endpoint_rows: list[str] = []
    matched_rows: list[str] = []
    for cell in adapter["cells"]:
        regime = str(cell["regime_id"])
        ra = cell["ra_historical_average_plateau"]
        append = cell["append"]
        append_endpoints = append["endpoints"]
        cost_cell = cost_cells[regime]
        ra_69 = cost_cell["ra_round_69"]
        common = cost_cell["common_accuracy"]
        endpoint_rows.append(
            " & ".join(
                (
                    str(cell["display_name"]),
                    format_delta_e(float(ra["points"][-1]["delta_e"])),
                    format_costs(ra_69["costs"]),
                    format_delta_e(float(append_endpoints["round_70"]["delta_e"])),
                    format_costs(append_endpoints["round_70"]["costs"]),
                )
            )
            + r" \\"
        )
        ra_common = common["ra"]
        append_common = common["append"]
        matched_rows.append(
            " & ".join(
                (
                    REGIME_ABBREVIATIONS[regime],
                    str(common["shared_window_end"]),
                    format_delta_e(float(common["target_delta_e"])),
                    str(ra_common["round"]),
                    format_delta_e(float(ra_common["delta_e"])),
                    format_costs(ra_common["costs"]),
                    str(append_common["round"]),
                    format_delta_e(float(append_common["delta_e"])),
                    format_costs(append_common["costs"]),
                )
            )
            + r" \\"
        )
    plot_argument = latex_escape(plot_pdf.resolve().as_posix())
    tex = rf"""\documentclass[10pt,letterpaper]{{article}}
\usepackage[landscape,margin=0.16in]{{geometry}}
\usepackage{{amsmath,booktabs,graphicx}}
\usepackage[T1]{{fontenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
\includegraphics[width=0.92\textwidth,height=2.85in,keepaspectratio]{{{plot_argument}}}
\vspace{{0.40em}}

\tiny
\setlength{{\tabcolsep}}{{2.6pt}}
\resizebox{{0.985\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\toprule
Regime & $\Delta E_{{69}}^{{\rm RA}}$ & $C_{{69}}^{{\rm RA}}$ &
$\Delta E_{{70}}^{{\rm Append}}$ & $C_{{70}}^{{\rm Append}}$ \\
\midrule
{chr(10).join(endpoint_rows)}
\bottomrule
\end{{tabular}}}}
\vspace{{0.15em}}

{{\scriptsize\bfseries Common-accuracy costs before the earlier effective plateau}}
\vspace{{-0.15em}}

\tiny
\setlength{{\tabcolsep}}{{2.15pt}}
\renewcommand{{\arraystretch}}{{0.76}}
\resizebox{{0.985\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}ccc r c c r c c@{{}}}}
\toprule
Reg. & $K_\cap$ & $\Delta E_\cap$ & $k_\cap^{{\rm RA}}$ &
$\Delta E_{{\rm RA}}$ & $C_{{\rm RA}}$ & $k_\cap^{{\rm Append}}$ &
$\Delta E_{{\rm Append}}$ & $C_{{\rm Append}}$ \\
\midrule
{chr(10).join(matched_rows)}
\bottomrule
\end{{tabular}}}}
\end{{center}}
\vspace{{-0.55em}}
\tiny
$C=(N_{{2q}},D_{{2q}},D_c,W_{{1q}},S_{{\rm alg}})$; all errors use exact
diagonalization at the identical phonon cutoff. Qiskit fields use the common
source-locked Table-I compiler (optimization level 0, seed 7, reference state
included). RA $S_{{\rm alg}}$ is unavailable because the EXDEV failure prevented
checkpoint/estimator-ledger transfer; no estimate is substituted. Append
$S_{{\rm alg}}$ is read from the signed prefix checkpoint. $K_\cap$ is the
earlier effective plateau and each method is costed at its earliest crossing of
$\Delta E_\cap$.
\end{{document}}
"""
    tex_path.write_text(tex, encoding="utf-8")


def build_assets(
    adapter: Mapping[str, Any],
    *,
    cost_adapter: Mapping[str, Any],
    asset_dir: Path,
    asset_stem: str,
) -> dict[str, Path]:
    if not ASSET_STEM_RE.fullmatch(asset_stem) or asset_stem in {".", ".."}:
        raise Page6InputError("asset_stem must be a safe filename component")
    assets = {
        "plot_png": asset_dir / f"{asset_stem}_plot.png",
        "plot_pdf": asset_dir / f"{asset_stem}_plot.pdf",
        "page_tex": asset_dir / f"{asset_stem}.tex",
        "page_pdf": asset_dir / f"{asset_stem}.pdf",
    }
    render_plot(adapter, png_path=assets["plot_png"], pdf_path=assets["plot_pdf"])
    write_page_tex(
        adapter,
        cost_adapter=cost_adapter,
        plot_pdf=assets["plot_pdf"],
        tex_path=assets["page_tex"],
    )
    legacy_page._compile_page(assets["page_tex"], assets["page_pdf"])
    return assets


def replace_page6(
    *,
    target_pdf: Path,
    target_provenance: Path,
    combined_adapter_path: Path,
    cost_adapter_path: Path,
    asset_dir: Path,
    asset_stem: str,
) -> dict[str, Any]:
    combined = load_json(combined_adapter_path, label="combined page-6 adapter")
    combined_sha256 = verify_self_digest(combined, label="combined page-6 adapter")
    if combined.get("schema") != COMBINED_ADAPTER_SCHEMA:
        raise Page6InputError("combined page-6 adapter schema drifted")
    cost_adapter = validate_cost_adapter(
        cost_adapter_path,
        combined=combined,
        combined_path=combined_adapter_path,
        combined_canonical_sha256=combined_sha256,
    )
    provenance = load_json(target_provenance, label="target provenance")
    output_binding = mapping(
        mapping(provenance.get("outputs"), label="outputs").get("partial_progress_pdf"),
        label="partial_progress_pdf",
    )
    if output_binding.get("sha256") != sha256_file(target_pdf):
        raise Page6InputError("target PDF/provenance binding drifted")
    layout = mapping(provenance.get("layout"), label="layout")
    before_hashes = legacy_page._page_content_hashes(target_pdf)
    page_count = len(before_hashes)
    if page_count < 6 or layout.get("page_count") != page_count:
        raise Page6InputError(
            "target report page count/provenance binding is invalid"
        )
    existing_comparison = provenance.get("ra_append_singleton_r70_comparison")
    if layout.get("page_6") == PAGE_ID:
        comparison = mapping(existing_comparison, label="existing comparison")
        adapter_binding = mapping(comparison.get("adapter"), label="adapter binding")
        cost_binding = mapping(
            comparison.get("cost_adapter"), label="cost adapter binding"
        )
        if (
            adapter_binding.get("canonical_sha256") != combined_sha256
            or cost_binding.get("canonical_sha256") != cost_adapter["sha256"]
        ):
            raise Page6InputError("page 6 already binds a different combined adapter")
        return {
            "status": "already_current",
            "output_pdf": str(target_pdf),
            "sha256": sha256_file(target_pdf),
            "pages": page_count,
        }
    if layout.get("page_6") not in {legacy_page.PAGE_ID, *PRIOR_PAGE_IDS}:
        raise Page6InputError("existing page-6 identity is unsupported")

    from pypdf import PdfReader, PdfWriter

    assets = build_assets(
        combined,
        cost_adapter=cost_adapter,
        asset_dir=asset_dir,
        asset_stem=asset_stem,
    )
    page_reader = PdfReader(str(assets["page_pdf"]), strict=False)
    if len(page_reader.pages) != 1:
        raise Page6InputError("combined page asset is not exactly one page")
    temporary_pdf = target_pdf.with_name(f".{target_pdf.name}.page6.tmp")
    writer = PdfWriter()
    existing_pages = PdfReader(str(target_pdf), strict=False).pages
    for page in existing_pages[:5]:
        writer.add_page(page)
    writer.add_page(page_reader.pages[0])
    for page in existing_pages[6:]:
        writer.add_page(page)
    try:
        with temporary_pdf.open("wb") as stream:
            writer.write(stream)
        after_hashes = legacy_page._page_content_hashes(temporary_pdf)
        if (
            len(after_hashes) != page_count
            or after_hashes[:5] != before_hashes[:5]
            or after_hashes[6:] != before_hashes[6:]
        ):
            raise Page6InputError("page replacement altered a non-target page")
        new_pdf_binding = file_binding(temporary_pdf)
        new_pdf_binding["path"] = str(target_pdf.resolve())
        updated = copy.deepcopy(provenance)
        if "prior_append_singleton_r70_progress" not in updated:
            updated["prior_append_singleton_r70_progress"] = copy.deepcopy(
                updated.get("append_singleton_r70_progress")
            )
        if "prior_ra_append_singleton_r70_comparison" not in updated:
            updated["prior_ra_append_singleton_r70_comparison"] = copy.deepcopy(
                updated.get("ra_append_singleton_r70_comparison")
            )
        updated["layout"]["page_6"] = PAGE_ID
        updated["outputs"]["partial_progress_pdf"] = new_pdf_binding
        for output_key, asset_key in (
            ("ra_append_singleton_r70_plot_png", "plot_png"),
            ("ra_append_singleton_r70_plot_pdf", "plot_pdf"),
            ("ra_append_singleton_r70_page_tex", "page_tex"),
            ("ra_append_singleton_r70_page_pdf", "page_pdf"),
        ):
            updated["outputs"][output_key] = file_binding(assets[asset_key])
        updated["append_singleton_r70_progress"] = {
            "schema": legacy_page.PAGE_ID,
            "status": "incorporated_into_combined_page_6",
            "completed_regimes": list(REGIME_ORDER),
            "pending_regimes": [],
            "current_page_id": PAGE_ID,
            "combined_provenance_key": "ra_append_singleton_r70_comparison",
            "adapter": copy.deepcopy(combined["append_adapter"]),
        }
        updated["ra_append_singleton_r70_comparison"] = {
            "schema": PAGE_ID,
            "classification": "supplemental_diagnostic_not_adopted_evidence",
            "page_id": PAGE_ID,
            "adapter": {
                **file_binding(combined_adapter_path),
                "canonical_sha256": combined_sha256,
            },
            "cost_adapter": {
                **copy.deepcopy(cost_adapter["file_binding"]),
                "canonical_sha256": cost_adapter["sha256"],
            },
            "append_completed_regimes": list(REGIME_ORDER),
            "ra_accepted_rounds_observed": 70,
            "ra_recoverable_energy_rounds": {"minimum": 0, "maximum": 69},
            "limitations": [*copy.deepcopy(combined["limitations"]), RA_S_ALG_LIMITATION],
            "cost_cells": copy.deepcopy(cost_adapter["cells"]),
            "cells": [
                {
                    "regime_id": cell["regime_id"],
                    "append_round_70": copy.deepcopy(
                        cell["append"]["endpoints"]["round_70"]
                    ),
                    "ra_round_69": copy.deepcopy(
                        cell["ra_historical_average_plateau"]["points"][-1]
                    ),
                    "ra_source": copy.deepcopy(
                        cell["ra_historical_average_plateau"]["source"]
                    ),
                }
                for cell in combined["cells"]
            ],
            "structural_validation": {
                "pages": page_count,
                "preserved_page_content_sha256": before_hashes[:5],
                "preserved_pages_7_onward_content_sha256": before_hashes[6:],
                "prior_page_6_content_sha256": before_hashes[5],
                "new_page_6_content_sha256": after_hashes[5],
            },
            "outputs": {
                key: copy.deepcopy(updated["outputs"][key])
                for key in (
                    "ra_append_singleton_r70_plot_png",
                    "ra_append_singleton_r70_plot_pdf",
                    "ra_append_singleton_r70_page_tex",
                    "ra_append_singleton_r70_page_pdf",
                )
            },
        }
        updated["limitations"] = [
            limitation
            for limitation in updated.get("limitations", [])
            if limitation != OLD_PAGE_LIMITATION
        ]
        if NEW_PAGE_LIMITATION not in updated["limitations"]:
            updated["limitations"].append(NEW_PAGE_LIMITATION)
        if RA_S_ALG_LIMITATION not in updated["limitations"]:
            updated["limitations"].append(RA_S_ALG_LIMITATION)
        os.replace(temporary_pdf, target_pdf)
        legacy_page._atomic_write_json(target_provenance, updated)
    finally:
        temporary_pdf.unlink(missing_ok=True)
    return {
        "status": "replaced_page_6",
        "output_pdf": str(target_pdf),
        "output_provenance": str(target_provenance),
        "sha256": sha256_file(target_pdf),
        "pages": page_count,
        "preserved_pages_1_5": True,
        "preserved_pages_7_onward": True,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--append-adapter", type=Path, required=True)
    result.add_argument("--ra-retrieval-root", type=Path, required=True)
    result.add_argument("--ra-package-dir", type=Path, required=True)
    result.add_argument("--combined-adapter", type=Path, required=True)
    result.add_argument("--cost-adapter", type=Path, required=True)
    result.add_argument("--target-pdf", type=Path, required=True)
    result.add_argument("--target-provenance", type=Path, required=True)
    result.add_argument("--asset-dir", type=Path, required=True)
    result.add_argument("--asset-stem", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        build_combined_adapter(
            append_adapter_path=args.append_adapter.resolve(),
            ra_retrieval_root=args.ra_retrieval_root.resolve(),
            ra_package_dir=args.ra_package_dir.resolve(),
            output=args.combined_adapter.resolve(),
        )
        result = replace_page6(
            target_pdf=args.target_pdf.resolve(),
            target_provenance=args.target_provenance.resolve(),
            combined_adapter_path=args.combined_adapter.resolve(),
            cost_adapter_path=args.cost_adapter.resolve(),
            asset_dir=args.asset_dir.resolve(),
            asset_stem=args.asset_stem,
        )
    except (OSError, Page6InputError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

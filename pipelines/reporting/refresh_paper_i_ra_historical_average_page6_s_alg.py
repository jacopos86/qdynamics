#!/usr/bin/env python3
"""Recover page-6 RA prefix costs from the publication-fixed v5 receipts.

This is a report-only repair.  It verifies that the v5 source archive and
source-lock snapshot match v4, checks the replacement trajectory against the
already plotted v4 energies and accepted choices, reads closed ``S_alg``
prefixes from each result receipt, and replaces only page 6 of the evolving
Paper-I diagnostic PDF.  The replacement remains supplemental diagnostic
material and does not adopt either run as Paper-I evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import (  # noqa: E402
    add_paper_i_historical_mean_global_singleton_full6_page as page_helpers,
)
from pipelines.reporting import (  # noqa: E402
    update_paper_i_ra_append_r70_singleton_page as page6_plot_builder,
)


REPORT_DIR = REPO_ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
)
REPORT_STEM = "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
MASTER_PDF = REPORT_DIR / f"{REPORT_STEM}_partial_progress.pdf"
MASTER_PROVENANCE = REPORT_DIR / f"{REPORT_STEM}_partial_progress_provenance.json"
TRAJECTORY_ADAPTER = REPORT_DIR / f"{REPORT_STEM}_ra_append_singleton_r70_page6_adapter.json"
LEGACY_COST_ADAPTER = REPORT_DIR / "paper_i_ra_append_singleton_r70_prefix_costs_v1.json"
OUTPUT_COST_ADAPTER = REPORT_DIR / "paper_i_ra_append_singleton_r70_prefix_costs_v2.json"
PAGE6_STEM = f"{REPORT_STEM}_partial_progress_page6_accuracy_and_closed_s_alg"
PAGE6_PLOT_PNG = REPORT_DIR / f"{PAGE6_STEM}_plot.png"
PAGE6_PLOT = REPORT_DIR / f"{PAGE6_STEM}_plot.pdf"
PAGE6_TEX = REPORT_DIR / f"{PAGE6_STEM}.tex"
PAGE6_PDF = REPORT_DIR / f"{PAGE6_STEM}.pdf"

V4_PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_average_singleton_plateau6_r70_fresh_20260801_v4_chtc"
)
V5_PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_average_singleton_plateau6_r70_fresh_20260802_v5_chtc"
)
V5_RETRIEVAL_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_chtc_20260803_historical_average_plateau_r70_cluster_9400878"
)

EXPECTED_TRAJECTORY_ADAPTER_SHA256 = (
    "81dabf09c64fb7bbd2c39f403d2bdb36f285339f1c2ff29870b04b19055b84a0"
)
EXPECTED_LEGACY_COST_ADAPTER_SHA256 = (
    "1d6de7977b30d7842b6ee1b73b066f5b92751ee4d16efcc55cb9595e4f5398d3"
)
EXPECTED_V4_MANIFEST_SHA256 = (
    "5bfa293ebcb467fb69b95b27dc675465887e4f846a9e94c821e4e56ae0906d96"
)
EXPECTED_V5_MANIFEST_SHA256 = (
    "f6e6e2b58220d3f94943e6d2a1e36e4ae1845ac894c980eddad289eaf7f8de20"
)
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "f8b42ea0411e9f3f763d79bcddb7ab39b1873550451e5a5e73cd53b88b07ec26"
)
EXPECTED_SOURCE_LOCKS_SHA256 = (
    "7415ae292fdb34bc1ecc57a37904bcccaa645491185a640ae0210de53add03e7"
)

PAGE_ID = "historical_average_r70_accuracy_and_closed_s_alg_v3"
RESULT_MEMBER = "worker_outputs/artifacts/result.json"
REGIME_ORDER = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
REGIME_ABBREVIATIONS = {
    "weak_weak": "WW",
    "intermediate_weak": "IW",
    "strong_weak_u8": "SW",
    "weak_strong": "WS",
    "intermediate_strong": "IS",
    "strong_strong_u8": "SS",
}
EXECUTIONS = {
    "weak_weak": (
        "historical_average_v5_r70_fresh__weak_weak__nph3__ra_singleton_plateau",
        0,
    ),
    "intermediate_weak": (
        "historical_average_v5_r70_fresh__intermediate_weak__nph3__ra_singleton_plateau",
        1,
    ),
    "strong_weak_u8": (
        "historical_average_v5_r70_fresh__strong_weak_u8__nph3__ra_singleton_plateau",
        2,
    ),
    "weak_strong": (
        "historical_average_v5_r70_fresh__weak_strong__nph7__ra_singleton_plateau",
        3,
    ),
    "intermediate_strong": (
        "historical_average_v5_r70_fresh__intermediate_strong__nph7__ra_singleton_plateau",
        4,
    ),
    "strong_strong_u8": (
        "historical_average_v5_r70_fresh__strong_strong_u8__nph7__ra_singleton_plateau",
        5,
    ),
}
COMPONENT_FIELDS = ("n_h_outer", "n_h_refit", "n_grad", "n_metric")
COST_FIELDS = ("N2q", "D2q", "Dc", "W1q", "S_alg")


class Page6RepairError(ValueError):
    """The report repair would weaken or misbind the page-6 evidence."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_digest(value: Mapping[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def _digested(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop("sha256", None)
    result["sha256"] = hashlib.sha256(_canonical_json_bytes(result)).hexdigest()
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Page6RepairError(f"{label} is missing or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Page6RepairError(f"{label} is not a JSON object")
    return value


def _verify_self_digest(
    value: Mapping[str, Any], *, expected: str, label: str
) -> None:
    if value.get("sha256") != expected or _canonical_digest(value) != expected:
        raise Page6RepairError(f"{label} canonical identity drifted")


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _binding(path: Path, *, relative: bool = False) -> dict[str, Any]:
    return {
        "path": _relative_path(path) if relative else str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
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
    finally:
        temporary.unlink(missing_ok=True)


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise Page6RepairError(f"{label} is not an integer")
    result = int(value)
    if result < minimum or result != value:
        raise Page6RepairError(f"{label} is outside its integer range")
    return result


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise Page6RepairError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Page6RepairError(f"{label} is not finite")
    return result


def _manifest_receipt(path: Path, *, expected: str, label: str) -> dict[str, Any]:
    manifest = _load(path, label=label)
    _verify_self_digest(manifest, expected=expected, label=label)
    source_archive = manifest.get("source_archive")
    source_locks = manifest.get("source_locks_snapshot")
    if not isinstance(source_archive, Mapping) or not isinstance(source_locks, Mapping):
        raise Page6RepairError(f"{label} lacks source-lock receipts")
    if (
        source_archive.get("sha256") != EXPECTED_SOURCE_ARCHIVE_SHA256
        or source_locks.get("canonical_sha256") != EXPECTED_SOURCE_LOCKS_SHA256
    ):
        raise Page6RepairError(f"{label} source identity drifted")
    return {
        **_binding(path, relative=True),
        "canonical_sha256": expected,
        "campaign_id": manifest.get("campaign_id"),
        "source_archive_sha256": source_archive.get("sha256"),
        "source_locks_canonical_sha256": source_locks.get("canonical_sha256"),
    }


def _extract_result_member(archive_path: Path) -> tuple[Path, dict[str, Any]]:
    temporary = tempfile.NamedTemporaryFile(prefix="page6-result-", suffix=".json", delete=False)
    temporary_path = Path(temporary.name)
    digest = hashlib.sha256()
    size_bytes = 0
    seen = False
    try:
        with temporary:
            with tarfile.open(archive_path, "r|gz") as archive:
                for member in archive:
                    if member.name != RESULT_MEMBER:
                        continue
                    if seen or not member.isfile() or member.issym() or member.islnk():
                        raise Page6RepairError("result member is duplicated or unsafe")
                    seen = True
                    source = archive.extractfile(member)
                    if source is None:
                        raise Page6RepairError("result member is unreadable")
                    while True:
                        chunk = source.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        temporary.write(chunk)
                        digest.update(chunk)
                        size_bytes += len(chunk)
        if not seen or size_bytes == 0:
            raise Page6RepairError(f"archive lacks {RESULT_MEMBER}")
        return temporary_path, {
            "path": RESULT_MEMBER,
            "sha256": digest.hexdigest(),
            "size_bytes": size_bytes,
        }
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _ijson_items(path: Path, prefix: str) -> list[Any]:
    try:
        import ijson
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise Page6RepairError("v5 receipt extraction requires ijson") from exc
    with path.open("rb") as stream:
        return list(ijson.items(stream, prefix, use_float=True))


def _single_ijson_item(path: Path, prefix: str, *, label: str) -> Any:
    values = _ijson_items(path, prefix)
    if len(values) != 1:
        raise Page6RepairError(f"{label} is not unique")
    return values[0]


def _component_receipt(work: Mapping[str, Any], *, label: str) -> dict[str, int]:
    components = work.get("components")
    if not isinstance(components, Mapping):
        raise Page6RepairError(f"{label} lacks S_alg components")
    raw = {
        field: _integer(components.get(field), label=f"{label} {field}")
        for field in COMPONENT_FIELDS
    }
    s_alg = _integer(work.get("s_alg"), label=f"{label} S_alg")
    if sum(raw.values()) != s_alg:
        raise Page6RepairError(f"{label} S_alg components do not close")
    return {
        "N_H_outer": raw["n_h_outer"],
        "N_H_refit": raw["n_h_refit"],
        "N_grad": raw["n_grad"],
        "N_metric": raw["n_metric"],
    }


def _validate_replacement_result(
    *,
    regime: str,
    result_path: Path,
    trajectory_cell: Mapping[str, Any],
) -> dict[str, Any]:
    trajectory = _ijson_items(result_path, "run.accepted_trajectory.item")
    transitions = _ijson_items(result_path, "run.accepted_transitions.item")
    prefix_work = _ijson_items(
        result_path, "run.canonical_reporting.accepted_prefix_work.item"
    )
    accounting = _single_ijson_item(
        result_path, "run.estimator_accounting", label=f"{regime} accounting"
    )
    if not isinstance(accounting, Mapping):
        raise Page6RepairError(f"{regime} accounting is not an object")
    if (
        accounting.get("complete") is not True
        or accounting.get("prefix_closure_passed") is not True
    ):
        raise Page6RepairError(f"{regime} estimator accounting is not closed")
    if not (len(trajectory) == len(transitions) == len(prefix_work) == 70):
        raise Page6RepairError(f"{regime} replacement horizon is not 70 rounds")

    ra = trajectory_cell.get("ra_historical_average_plateau")
    if not isinstance(ra, Mapping):
        raise Page6RepairError(f"{regime} trajectory adapter lacks RA data")
    points = {
        _integer(point.get("round"), label=f"{regime} point round"): point
        for point in ra.get("points", ())
        if isinstance(point, Mapping)
    }
    decisions = {
        _integer(decision.get("accepted_round"), label=f"{regime} decision round"): decision
        for decision in ra.get("accepted_decisions", ())
        if isinstance(decision, Mapping)
    }
    if set(points) != set(range(70)) or set(decisions) != set(range(1, 71)):
        raise Page6RepairError(f"{regime} original trajectory closure drifted")

    compact_work: list[dict[str, Any]] = []
    compact_transitions: list[dict[str, Any]] = []
    for index, (state, transition, work) in enumerate(
        zip(trajectory, transitions, prefix_work, strict=True), start=1
    ):
        if not all(isinstance(value, Mapping) for value in (state, transition, work)):
            raise Page6RepairError(f"{regime} round {index} receipt is malformed")
        if (
            state.get("controller_round") != index
            or transition.get("controller_round") != index
        ):
            raise Page6RepairError(f"{regime} replacement round ordering drifted")
        energy = _finite(state.get("energy"), label=f"{regime} round {index} energy")
        if transition.get("energy_after") != energy:
            raise Page6RepairError(f"{regime} transition energy does not close")
        decision = decisions[index]
        if (
            transition.get("selected_operator") != decision.get("candidate_label")
            or transition.get("insertion_position") != decision.get("selected_position")
        ):
            raise Page6RepairError(f"{regime} accepted choice differs at round {index}")
        if index <= 69 and points[index].get("energy") != energy:
            raise Page6RepairError(f"{regime} plotted energy differs at round {index}")
        components = _component_receipt(work, label=f"{regime} round {index}")
        s_alg = sum(components.values())
        if transition.get("cumulative_s_alg") != s_alg:
            raise Page6RepairError(f"{regime} transition S_alg does not close")
        compact_work.append({"s_alg": s_alg, "components": components})
        compact_transitions.append(
            {
                "estimator_prefix_after": transition.get("estimator_prefix_after"),
                "ledger_closure_sha256": transition.get("ledger_closure_sha256"),
            }
        )
    return {
        "prefix_work": compact_work,
        "prefix_transitions": compact_transitions,
        "validation": {
            "accepted_rounds": 70,
            "plotted_energy_rounds_matched_exactly": [1, 69],
            "accepted_choice_rounds_matched_exactly": [1, 70],
            "accepted_choice_fields": ["selected_operator", "insertion_position"],
            "estimator_accounting_complete": True,
            "prefix_closure_passed": True,
        },
    }


def _fill_observation(
    observation: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    execution_id: str,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(observation))
    round_index = _integer(result.get("round"), label="cost observation round", minimum=1)
    prefix_work = receipt["prefix_work"][round_index - 1]
    transition = receipt["prefix_transitions"][round_index - 1]
    costs = result.get("costs")
    if not isinstance(costs, dict):
        raise Page6RepairError("cost observation lacks a mutable cost tuple")
    costs["S_alg"] = prefix_work["s_alg"]
    result["S_alg_status"] = "available_closed_prefix_receipt_v5"
    result["S_alg_components"] = copy.deepcopy(prefix_work["components"])
    result["replacement_prefix_receipt"] = {
        "execution_id": execution_id,
        "controller_round": round_index,
        "estimator_prefix_after": transition["estimator_prefix_after"],
        "ledger_closure_sha256": transition["ledger_closure_sha256"],
    }
    return result


def build_cost_adapter(*, write: bool) -> dict[str, Any]:
    trajectory = _load(TRAJECTORY_ADAPTER, label="page-6 trajectory adapter")
    legacy = _load(LEGACY_COST_ADAPTER, label="legacy page-6 cost adapter")
    _verify_self_digest(
        trajectory,
        expected=EXPECTED_TRAJECTORY_ADAPTER_SHA256,
        label="page-6 trajectory adapter",
    )
    _verify_self_digest(
        legacy,
        expected=EXPECTED_LEGACY_COST_ADAPTER_SHA256,
        label="legacy page-6 cost adapter",
    )
    if tuple(trajectory.get("regime_order", ())) != REGIME_ORDER or tuple(
        legacy.get("regime_order", ())
    ) != REGIME_ORDER:
        raise Page6RepairError("page-6 regime order drifted")

    v4_manifest = V4_PACKAGE_DIR / "package_manifest.json"
    v5_manifest = V5_PACKAGE_DIR / "package_manifest.json"
    v4_receipt = _manifest_receipt(
        v4_manifest, expected=EXPECTED_V4_MANIFEST_SHA256, label="v4 package manifest"
    )
    v5_receipt = _manifest_receipt(
        v5_manifest, expected=EXPECTED_V5_MANIFEST_SHA256, label="v5 package manifest"
    )
    if (
        v4_receipt["source_archive_sha256"] != v5_receipt["source_archive_sha256"]
        or v4_receipt["source_locks_canonical_sha256"]
        != v5_receipt["source_locks_canonical_sha256"]
    ):
        raise Page6RepairError("v4/v5 source-lock equivalence failed")

    trajectory_cells = {
        str(cell["regime_id"]): cell for cell in trajectory.get("cells", ())
    }
    result = copy.deepcopy(legacy)
    result.pop("sha256", None)
    result["schema"] = "paper_i_ra_append_singleton_r70_cost_diagnostic_v2"
    result["status"] = "passed_with_closed_v5_ra_prefix_receipts"
    result["classification"] = "supplemental_diagnostic_not_adopted_evidence"
    result["sources"] = {
        "ra_original_trajectory_package": v4_receipt,
        "ra_prefix_receipt_package": v5_receipt,
        "append_package": copy.deepcopy(legacy["sources"]["append_package"]),
        "same_cutoff_reference": copy.deepcopy(
            legacy["sources"]["same_cutoff_reference"]
        ),
        "source_lock_equivalence": {
            "status": "passed",
            "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
            "source_locks_canonical_sha256": EXPECTED_SOURCE_LOCKS_SHA256,
        },
    }
    result["limitations"] = [
        "The publication-fixed v5 receipts complete S_alg for this supplemental "
        "diagnostic only; neither the v4 trace nor the v5 replacement is adopted "
        "Paper-I evidence by this report update."
    ]

    cost_cells = {str(cell["regime_id"]): cell for cell in result.get("cells", ())}
    replacement_rows: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        execution_id, proc_id = EXECUTIONS[regime]
        archive_path = V5_RETRIEVAL_DIR / "fetched" / (
            f"{execution_id}__cluster_9400878__proc_{proc_id}.tar.gz"
        )
        if not archive_path.is_file() or archive_path.is_symlink():
            raise Page6RepairError(f"{regime} v5 result archive is unavailable")
        archive_binding = _binding(archive_path, relative=True)
        result_path, result_member = _extract_result_member(archive_path)
        try:
            receipt = _validate_replacement_result(
                regime=regime,
                result_path=result_path,
                trajectory_cell=trajectory_cells[regime],
            )
        finally:
            result_path.unlink(missing_ok=True)

        cell = cost_cells[regime]
        cell["ra_round_69"] = _fill_observation(
            cell["ra_round_69"], receipt=receipt, execution_id=execution_id
        )
        cell["common_accuracy"]["ra"] = _fill_observation(
            cell["common_accuracy"]["ra"],
            receipt=receipt,
            execution_id=execution_id,
        )
        cell["replacement_receipt"] = {
            "cluster_id": 9400878,
            "proc_id": proc_id,
            "execution_id": execution_id,
            "result_archive": archive_binding,
            "result_member": result_member,
            "trajectory_equivalence": receipt["validation"],
        }
        replacement_rows.append(
            {
                "regime_id": regime,
                "execution_id": execution_id,
                "result_archive_sha256": archive_binding["sha256"],
                "result_member_sha256": result_member["sha256"],
                **receipt["validation"],
            }
        )

    result["replacement_validation"] = {
        "status": "passed",
        "original_cluster_id": 9400249,
        "replacement_cluster_id": 9400878,
        "source_lock_equivalence": "exact",
        "cells": replacement_rows,
    }
    digested = _digested(result)
    if OUTPUT_COST_ADAPTER.exists():
        existing = _load(OUTPUT_COST_ADAPTER, label="existing v2 cost adapter")
        if existing.get("sha256") != digested["sha256"] or _canonical_digest(
            existing
        ) != digested["sha256"]:
            raise Page6RepairError("refusing to replace a different v2 cost adapter")
    elif write:
        _atomic_write_json(OUTPUT_COST_ADAPTER, digested)
    return digested


def load_validated_cost_adapter() -> dict[str, Any]:
    result = _load(OUTPUT_COST_ADAPTER, label="v2 cost adapter")
    if (
        result.get("schema") != "paper_i_ra_append_singleton_r70_cost_diagnostic_v2"
        or result.get("status") != "passed_with_closed_v5_ra_prefix_receipts"
        or result.get("classification")
        != "supplemental_diagnostic_not_adopted_evidence"
        or result.get("sha256") != _canonical_digest(result)
        or tuple(result.get("regime_order", ())) != REGIME_ORDER
    ):
        raise Page6RepairError("v2 cost adapter identity or closure drifted")
    return result


def _latex_escape(value: Any) -> str:
    return page_helpers.latex_escape(str(value))


def _sci(value: Any) -> str:
    number = _finite(value, label="displayed energy error")
    if number == 0.0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(number))))
    mantissa = number / (10.0**exponent)
    return f"${mantissa:.2f}\\mathord{{\\times}}10^{{{exponent}}}$"


def _cost5(costs: Mapping[str, Any]) -> str:
    return "(" + ", ".join(
        f"{_integer(costs.get(field), label=f'displayed {field}'):,}"
        for field in COST_FIELDS
    ) + ")"


def write_page_tex(cost_adapter: Mapping[str, Any]) -> None:
    trajectory = _load(TRAJECTORY_ADAPTER, label="page-6 trajectory adapter")
    cost_cells = {
        str(cell["regime_id"]): cell for cell in cost_adapter.get("cells", ())
    }
    endpoint_rows: list[str] = []
    crossing_rows: list[str] = []
    for cell in trajectory["cells"]:
        regime = str(cell["regime_id"])
        costs = cost_cells[regime]
        ra = costs["ra_round_69"]
        append = cell["append"]["endpoints"]["round_70"]
        endpoint_rows.append(
            " & ".join(
                (
                    _latex_escape(cell["display_name"]),
                    _sci(ra["delta_e"]),
                    _latex_escape(_cost5(ra["costs"])),
                    _sci(append["delta_e"]),
                    _latex_escape(_cost5(append["costs"])),
                )
            )
            + r" \\"
        )
        common = costs["common_accuracy"]
        crossing_rows.append(
            " & ".join(
                (
                    REGIME_ABBREVIATIONS[regime],
                    str(common["shared_window_end"]),
                    _sci(common["target_delta_e"]),
                    str(common["ra"]["round"]),
                    _sci(common["ra"]["delta_e"]),
                    _latex_escape(_cost5(common["ra"]["costs"])),
                    str(common["append"]["round"]),
                    _sci(common["append"]["delta_e"]),
                    _latex_escape(_cost5(common["append"]["costs"])),
                )
            )
            + r" \\"
        )
    plot_path = _latex_escape(PAGE6_PLOT.resolve().as_posix())
    tex = rf"""\documentclass[10pt,letterpaper]{{article}}
\usepackage[landscape,margin=0.18in]{{geometry}}
\usepackage{{amsmath,booktabs,graphicx}}
\usepackage[T1]{{fontenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
{{\large\bfseries Historical-average global-singleton R70 diagnostic}}\\[-0.1ex]
{{\small The publication-fixed replacement reproduces the plotted RA trajectory and closes $S_{{\rm alg}}$ at every displayed prefix.}}

\includegraphics[width=0.91\textwidth,height=2.80in,keepaspectratio]{{{plot_path}}}
\vspace{{0.18em}}

\tiny
\setlength{{\tabcolsep}}{{2.6pt}}
\resizebox{{0.985\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\toprule
Regime & $|\Delta E_{{69}}^{{\rm RA}}|$ & $C_{{69}}^{{\rm RA}}$ &
$|\Delta E_{{70}}^{{\rm Append}}|$ & $C_{{70}}^{{\rm Append}}$ \\
\midrule
{chr(10).join(endpoint_rows)}
\bottomrule
\end{{tabular}}}}
\vspace{{0.16em}}

{{\scriptsize\bfseries Common-accuracy costs before the earlier effective plateau}}
\vspace{{-0.10em}}

\tiny
\setlength{{\tabcolsep}}{{2.1pt}}
\renewcommand{{\arraystretch}}{{0.76}}
\resizebox{{0.985\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}ccc r c c r c c@{{}}}}
\toprule
Reg. & $K_\cap$ & $|\Delta E_\cap|$ & $k_\cap^{{\rm RA}}$ &
$|\Delta E_{{\rm RA}}|$ & $C_{{\rm RA}}$ & $k_\cap^{{\rm Append}}$ &
$|\Delta E_{{\rm Append}}|$ & $C_{{\rm Append}}$ \\
\midrule
{chr(10).join(crossing_rows)}
\bottomrule
\end{{tabular}}}}
\end{{center}}
\vspace{{-0.35em}}
\tiny
$C=(N_{{2q}},D_{{2q}},D_c,W_{{1q}},S_{{\rm alg}})$. All errors use exact
diagonalization at the identical phonon cutoff. RA $S_{{\rm alg}}$ is read
from closed v5 prefix receipts. The v5 source archive and source-lock snapshot
match v4, and all plotted energies plus accepted operator/position choices
through $k=69$ match exactly. These replacement receipts remain supplemental
diagnostic material and are not adopted Paper-I evidence.
\end{{document}}
"""
    PAGE6_TEX.write_text(tex, encoding="utf-8")


def render_page_plot() -> None:
    trajectory = _load(TRAJECTORY_ADAPTER, label="page-6 trajectory adapter")
    _verify_self_digest(
        trajectory,
        expected=EXPECTED_TRAJECTORY_ADAPTER_SHA256,
        label="page-6 trajectory adapter",
    )
    page6_plot_builder.render_plot(
        trajectory,
        png_path=PAGE6_PLOT_PNG,
        pdf_path=PAGE6_PLOT,
    )


def compile_page() -> None:
    process = subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            PAGE6_TEX.name,
        ],
        cwd=REPORT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0 or not PAGE6_PDF.is_file():
        raise Page6RepairError(
            f"LaTeX build failed:\n{process.stdout}\n{process.stderr}"
        )
    subprocess.run(
        ["latexmk", "-c", PAGE6_TEX.name],
        cwd=REPORT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )


def _page_content_hashes(path: Path) -> list[str]:
    from pypdf import PdfReader

    hashes: list[str] = []
    for page in PdfReader(str(path), strict=False).pages:
        contents = page.get_contents()
        data = b"" if contents is None else contents.get_data()
        hashes.append(hashlib.sha256(data).hexdigest())
    return hashes


def replace_page6(cost_adapter: Mapping[str, Any]) -> dict[str, Any]:
    from pypdf import PdfReader, PdfWriter

    provenance = _load(MASTER_PROVENANCE, label="master provenance")
    current_binding = provenance.get("outputs", {}).get("partial_progress_pdf")
    if not isinstance(current_binding, Mapping) or current_binding.get("sha256") != _sha256(
        MASTER_PDF
    ):
        raise Page6RepairError("master PDF/provenance binding drifted")
    before_hashes = _page_content_hashes(MASTER_PDF)
    page_count = len(before_hashes)
    if page_count < 6 or provenance.get("layout", {}).get("page_count") != page_count:
        raise Page6RepairError("master page count/provenance binding drifted")
    replacement = PdfReader(str(PAGE6_PDF), strict=False)
    if len(replacement.pages) != 1:
        raise Page6RepairError("replacement page asset is not one page")

    writer = PdfWriter()
    original = PdfReader(str(MASTER_PDF), strict=False)
    for page in original.pages[:5]:
        writer.add_page(page)
    writer.add_page(replacement.pages[0])
    for page in original.pages[6:]:
        writer.add_page(page)
    if original.metadata:
        writer.add_metadata(dict(original.metadata))

    temporary_pdf = MASTER_PDF.with_name(f".{MASTER_PDF.name}.page6-salg.tmp")
    temporary_provenance = MASTER_PROVENANCE.with_name(
        f".{MASTER_PROVENANCE.name}.page6-salg.tmp"
    )
    rollback_pdf = MASTER_PDF.with_name(f".{MASTER_PDF.name}.page6-salg.rollback")
    for path in (temporary_pdf, temporary_provenance, rollback_pdf):
        path.unlink(missing_ok=True)
    with temporary_pdf.open("xb") as stream:
        writer.write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    after_hashes = _page_content_hashes(temporary_pdf)
    if (
        len(after_hashes) != page_count
        or after_hashes[:5] != before_hashes[:5]
        or after_hashes[6:] != before_hashes[6:]
    ):
        raise Page6RepairError("page replacement altered a non-target page")

    updated = copy.deepcopy(provenance)
    adapter_binding = {
        **_binding(OUTPUT_COST_ADAPTER),
        "canonical_sha256": cost_adapter["sha256"],
    }
    page_tex_binding = _binding(PAGE6_TEX)
    page_pdf_binding = _binding(PAGE6_PDF)
    plot_png_binding = _binding(PAGE6_PLOT_PNG)
    plot_pdf_binding = _binding(PAGE6_PLOT)
    updated["layout"]["page_6"] = PAGE_ID
    updated["outputs"]["partial_progress_pdf"] = {
        **_binding(temporary_pdf),
        "path": str(MASTER_PDF.resolve()),
    }
    updated["outputs"]["historical_average_r70_s_alg_cost_adapter"] = adapter_binding
    updated["outputs"]["historical_average_r70_s_alg_page_tex"] = page_tex_binding
    updated["outputs"]["historical_average_r70_s_alg_page_pdf"] = page_pdf_binding
    updated["outputs"]["historical_average_r70_s_alg_plot_png"] = plot_png_binding
    updated["outputs"]["historical_average_r70_s_alg_plot_pdf"] = plot_pdf_binding

    comparison = updated.get("ra_append_singleton_r70_comparison")
    if not isinstance(comparison, dict):
        raise Page6RepairError("page-6 comparison provenance is unavailable")
    comparison["status"] = "passed_with_closed_v5_ra_prefix_receipts"
    comparison["classification"] = "supplemental_diagnostic_not_adopted_evidence"
    comparison["cost_adapter"] = copy.deepcopy(adapter_binding)
    comparison["cost_cells"] = copy.deepcopy(cost_adapter["cells"])
    comparison["ra_prefix_receipt_source"] = copy.deepcopy(
        cost_adapter["sources"]["ra_prefix_receipt_package"]
    )
    comparison["limitations"] = [
        limitation
        for limitation in comparison.get("limitations", [])
        if "S_alg is unavailable" not in str(limitation)
        and "round 70 was accepted" not in str(limitation)
    ]
    comparison["limitations"].append(cost_adapter["limitations"][0])
    comparison["structural_validation"].update(
        {
            "prior_page_6_content_sha256": before_hashes[5],
            "new_page_6_content_sha256": after_hashes[5],
            "preserved_page_content_sha256": before_hashes[:5] + before_hashes[6:],
        }
    )

    completion = updated.get("historical_mean_global_singleton_cost_completion")
    if isinstance(completion, dict):
        completion.setdefault("outputs", {})["page6_pdf"] = page_pdf_binding
        completion["outputs"]["page6_tex"] = page_tex_binding
        completion["outputs"]["page6_plot_png"] = plot_png_binding
        completion["outputs"]["page6_plot_pdf"] = plot_pdf_binding
        completion["page6_presentation"] = (
            "Accuracy and closed RA/Append prefix costs share page 6."
        )

    updated["limitations"] = [
        limitation
        for limitation in updated.get("limitations", [])
        if not str(limitation).startswith("Page 6 compares complete authenticated")
        and not str(limitation).startswith("RA S_alg is unavailable")
    ]
    updated["limitations"].append(
        "Page 6 retains the original v4 accuracy curve through k=69 and fills "
        "RA S_alg from source-locked, publication-fixed v5 replacement receipts. "
        "The plotted energies and accepted choices match exactly; the page remains "
        "a supplemental diagnostic and is not adopted Paper-I evidence."
    )
    updated["historical_average_r70_s_alg_completion"] = {
        "schema": "paper_i_historical_average_r70_s_alg_completion_v1",
        "status": "passed",
        "classification": "supplemental_diagnostic_not_adopted_evidence",
        "cost_adapter": copy.deepcopy(adapter_binding),
        "source_lock_equivalence": copy.deepcopy(
            cost_adapter["sources"]["source_lock_equivalence"]
        ),
        "replacement_validation": copy.deepcopy(
            cost_adapter["replacement_validation"]
        ),
        "structural_validation": {
            "page_count": page_count,
            "replaced_page": 6,
            "prior_page_content_sha256": before_hashes[5],
            "new_page_content_sha256": after_hashes[5],
            "preserved_page_content_sha256": before_hashes[:5] + before_hashes[6:],
        },
        "outputs": {
            "page_tex": page_tex_binding,
            "page_pdf": page_pdf_binding,
            "plot_png": plot_png_binding,
            "plot_pdf": plot_pdf_binding,
        },
    }

    try:
        with temporary_provenance.open("xb") as stream:
            stream.write(
                json.dumps(updated, indent=2, sort_keys=True, allow_nan=False).encode(
                    "utf-8"
                )
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(MASTER_PDF, rollback_pdf)
        os.replace(temporary_pdf, MASTER_PDF)
        try:
            os.replace(temporary_provenance, MASTER_PROVENANCE)
        except Exception:
            os.replace(rollback_pdf, MASTER_PDF)
            raise
        rollback_pdf.unlink(missing_ok=True)
    except Exception:
        temporary_pdf.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)
        rollback_pdf.unlink(missing_ok=True)
        raise
    return {
        "status": "replaced_page_6_with_closed_s_alg",
        "pages": page_count,
        "pdf_sha256": _sha256(MASTER_PDF),
        "cost_adapter_sha256": cost_adapter["sha256"],
        "preserved_other_pages": True,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--reuse-validated-cost-adapter", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        cost_adapter = (
            load_validated_cost_adapter()
            if args.reuse_validated_cost_adapter
            else build_cost_adapter(write=not args.dry_run)
        )
        if args.dry_run:
            result = {
                "status": "validated_without_writes",
                "cost_adapter_sha256": cost_adapter["sha256"],
                "costs": [
                    {
                        "regime_id": cell["regime_id"],
                        "round_69_s_alg": cell["ra_round_69"]["costs"]["S_alg"],
                        "common_round": cell["common_accuracy"]["ra"]["round"],
                        "common_s_alg": cell["common_accuracy"]["ra"]["costs"][
                            "S_alg"
                        ],
                    }
                    for cell in cost_adapter["cells"]
                ],
            }
        else:
            render_page_plot()
            write_page_tex(cost_adapter)
            compile_page()
            result = replace_page6(cost_adapter)
    except (OSError, RuntimeError, Page6RepairError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

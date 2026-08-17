#!/usr/bin/env python3
"""Replace page 7 of the evolving RA/Append report with |Delta E| vs S_alg.

The plotted abscissae come only from authenticated cumulative estimator-ledger
receipts.  For live RA trajectories whose energy telemetry extends beyond the
last closed checkpoint, the curve stops at that checkpoint; no cost is inferred
for the stdout-only tail.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
)
MASTER_PDF = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress.pdf"
)
MASTER_PROVENANCE = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress_provenance.json"
)
SOURCE_ADAPTER = REPORT_DIR / (
    "historical_mean_global_singleton_live_page7_cost_filled_20260807_v3_adapter.json"
)
APPEND_R70_ADAPTER = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "append_singleton_r70_all6_adapter.json"
)
OUTPUT_STEM = (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "page7_deltae_vs_salg"
)
CURVE_JSON = REPORT_DIR / f"{OUTPUT_STEM}_curves.json"
PAGE_PDF = REPORT_DIR / f"{OUTPUT_STEM}.pdf"
PAGE_PNG = REPORT_DIR / f"{OUTPUT_STEM}.png"

REGIME_ORDER = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
PLOT_FLOOR = 1.0e-16
APPEND_COLOR = "#4C78A8"
RA_COLOR = "#E45756"

APPEND_MEMBER = "worker_outputs/payload/checkpoint.json"
APPEND_PREFIX = (
    "controller_replay_evidence.signed_controller_round_prefixes.item."
    "active_prefix_checkpoint.estimator_prefix.cumulative_executed_queries.S_alg"
)
RA_COMPLETE_MEMBER = "worker_outputs/artifacts/checkpoint.json"
RA_LIVE_MEMBER = "checkpoint.json"
RA_PREFIX = (
    "adapt_vqe.active_prefix_checkpoints.item.estimator_ledger_receipt."
    "cumulative_executed_queries.S_alg"
)
APPEND_R70_SCHEMA = "paper_i_append_adapt_singleton_r70_progress_adapter_v1"
APPEND_R70_PACKAGE_ID = (
    "paper_i_append_adapt_stationary_core12_r70_fresh_20260731_v1_chtc"
)
PAPER_FACING_COST_ROUND = 50
APPEND_TRAJECTORY_ROUND = 70


class PageUpdateError(ValueError):
    """The page cannot be updated without weakening its source contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PageUpdateError(f"missing or unsafe input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PageUpdateError(f"JSON input is not an object: {path}")
    return value


def canonical_digest(value: Mapping[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    payload = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def merge_append_r70(
    adapter: Mapping[str, Any],
    append_r70: Mapping[str, Any],
    *,
    append_r70_path: Path = APPEND_R70_ADAPTER,
) -> dict[str, Any]:
    """Overlay the authenticated k=0..70 Append curves onto page 7.

    The energy/S_alg curve is diagnostic through k=70.  The fixed cost
    observation remains the authenticated round-50 endpoint used elsewhere
    in the Paper-I report.
    """

    if append_r70.get("sha256") != canonical_digest(append_r70):
        raise PageUpdateError("Append R70 adapter self-digest drifted")
    raw_cells = append_r70.get("cells")
    if (
        append_r70.get("schema") != APPEND_R70_SCHEMA
        or append_r70.get("status") != "passed"
        or append_r70.get("package_id") != APPEND_R70_PACKAGE_ID
        or tuple(append_r70.get("regime_order", ())) != REGIME_ORDER
        or tuple(append_r70.get("completed_regimes", ())) != REGIME_ORDER
        or tuple(append_r70.get("pending_regimes", ())) != ()
        or not isinstance(raw_cells, list)
        or len(raw_cells) != len(REGIME_ORDER)
    ):
        raise PageUpdateError("Append R70 adapter identity drifted")
    r70_by_regime = {str(cell.get("regime_id")): cell for cell in raw_cells}
    if tuple(r70_by_regime) != REGIME_ORDER:
        raise PageUpdateError("Append R70 adapter regime closure drifted")

    merged = copy.deepcopy(dict(adapter))
    cells = merged.get("cells")
    if not isinstance(cells, list):
        raise PageUpdateError("page-7 source adapter has no cells")
    by_regime = {str(cell.get("regime_id")): cell for cell in cells}
    if tuple(by_regime) != REGIME_ORDER:
        raise PageUpdateError("page-7 source adapter regime order drifted")
    for regime in REGIME_ORDER:
        target = by_regime[regime]
        append = target.get("append")
        source = r70_by_regime[regime]
        if not isinstance(append, dict):
            raise PageUpdateError(f"{regime}: page-7 Append arm is invalid")
        if append.get("execution_id") != source.get("execution_id"):
            raise PageUpdateError(f"{regime}: Append execution identity drifted")
        if not math.isclose(
            float(append.get("exact_same_cutoff_energy")),
            float(source.get("exact_same_cutoff_energy")),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise PageUpdateError(f"{regime}: Append exact reference drifted")
        points = source.get("points")
        if not isinstance(points, list) or [
            int(point.get("round", -1)) for point in points
        ] != list(range(APPEND_TRAJECTORY_ROUND + 1)):
            raise PageUpdateError(f"{regime}: Append R70 curve is not k=0..70")
        endpoints = source.get("endpoints")
        if not isinstance(endpoints, Mapping):
            raise PageUpdateError(f"{regime}: Append endpoints are invalid")
        fixed = endpoints.get("round_50")
        trajectory = endpoints.get("round_70")
        if (
            not isinstance(fixed, Mapping)
            or int(fixed.get("round", -1)) != PAPER_FACING_COST_ROUND
            or not isinstance(trajectory, Mapping)
            or int(trajectory.get("round", -1)) != APPEND_TRAJECTORY_ROUND
            or not math.isclose(
                float(fixed.get("delta_e")),
                float(points[PAPER_FACING_COST_ROUND].get("delta_e")),
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
            or not math.isclose(
                float(trajectory.get("delta_e")),
                float(points[APPEND_TRAJECTORY_ROUND].get("delta_e")),
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
        ):
            raise PageUpdateError(f"{regime}: Append endpoint binding drifted")
        fixed_costs = fixed.get("costs")
        if not isinstance(fixed_costs, Mapping) or any(
            fixed_costs.get(field) is None
            for field in ("N2q", "D2q", "Dc", "W1q", "S_alg")
        ):
            raise PageUpdateError(f"{regime}: Append k=50 cost tuple is absent")
        append["points"] = copy.deepcopy(points)
        append["terminal"] = copy.deepcopy(dict(fixed))
        append["trajectory_terminal"] = {
            "round": APPEND_TRAJECTORY_ROUND,
            "energy": float(trajectory.get("energy")),
            "delta_e": float(trajectory.get("delta_e")),
        }
        append["source"] = copy.deepcopy(source.get("source"))
    merged["append_r70_adapter"] = binding(append_r70_path)
    merged["append_trajectory_round"] = APPEND_TRAJECTORY_ROUND
    merged["paper_facing_cost_round"] = PAPER_FACING_COST_ROUND
    return merged


def resolve_source_path(value: str) -> Path:
    candidate = Path(value)
    path = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    if not path.is_file() or path.is_symlink():
        raise PageUpdateError(f"source archive is missing or unsafe: {path}")
    return path.resolve()


def validate_archive_size(path: Path, source: Mapping[str, Any]) -> None:
    expected_size = source.get("size_bytes")
    if expected_size is not None and int(expected_size) != path.stat().st_size:
        raise PageUpdateError(f"source archive size drifted: {path}")


def locate_ra_complete_archive(cell: Mapping[str, Any]) -> Path:
    source = cell["ra"]["source"]
    execution_id = str(cell["ra"]["execution_id"])
    filename = (
        f"{execution_id}__cluster_{int(source['cluster_id'])}__"
        f"proc_{int(source['proc_id'])}.tar.gz"
    )
    search_root = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
    matches = [path.resolve() for path in search_root.rglob(filename) if path.is_file()]
    if len(matches) != 1:
        raise PageUpdateError(
            f"expected one local completed RA archive for {cell['regime_id']}, found {matches}"
        )
    validate_archive_size(matches[0], source["archive"])
    return matches[0]


def extract_s_alg(path: Path, *, member_name: str, prefix: str) -> list[int]:
    try:
        import ijson
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise PageUpdateError("S_alg extraction requires ijson") from exc

    print(f"extracting S_alg: {path.name} :: {member_name}", flush=True)
    member_seen = False
    values: list[int] = []
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.name != member_name:
                continue
            if member_seen or not member.isfile():
                raise PageUpdateError(f"archive member is duplicated or unsafe: {member_name}")
            member_seen = True
            stream = archive.extractfile(member)
            if stream is None:
                raise PageUpdateError(f"archive member is unreadable: {member_name}")
            values = [int(value) for value in ijson.items(stream, prefix, use_float=True)]
            break
    if not member_seen:
        raise PageUpdateError(f"archive member is missing: {path}#{member_name}")
    if not values or any(value <= 0 for value in values):
        raise PageUpdateError(f"no positive cumulative S_alg curve in {path}#{member_name}")
    if any(right <= left for left, right in zip(values, values[1:], strict=False)):
        raise PageUpdateError(f"S_alg is not strictly increasing in {path}#{member_name}")
    print(f"  recovered {len(values)} prefixes; terminal S_alg={values[-1]:,}", flush=True)
    return values


def curve_from_points(
    method: Mapping[str, Any],
    s_alg_values: Sequence[int],
    *,
    expected_terminal_s_alg: int,
    expected_s_alg_round: int | None = None,
    label: str,
) -> list[dict[str, Any]]:
    points = method.get("points")
    if not isinstance(points, list) or len(points) < len(s_alg_values) + 1:
        raise PageUpdateError(f"{label} has fewer energy points than cost prefixes")
    selected = points[: len(s_alg_values) + 1]
    expected_rounds = list(range(len(selected)))
    observed_rounds = [int(point["round"]) for point in selected]
    if observed_rounds != expected_rounds:
        raise PageUpdateError(f"{label} rounds are not a contiguous prefix")
    anchor_round = (
        len(s_alg_values) if expected_s_alg_round is None else expected_s_alg_round
    )
    if not 1 <= anchor_round <= len(s_alg_values):
        raise PageUpdateError(f"{label} S_alg anchor round is outside the curve")
    if int(s_alg_values[anchor_round - 1]) != int(expected_terminal_s_alg):
        raise PageUpdateError(
            f"{label} S_alg does not close to its authenticated round-{anchor_round} endpoint"
        )
    result = [
        {
            "round": 0,
            "S_alg": 0,
            "delta_e": float(selected[0]["delta_e"]),
        }
    ]
    result.extend(
        {
            "round": int(point["round"]),
            "S_alg": int(s_alg),
            "delta_e": float(point["delta_e"]),
        }
        for point, s_alg in zip(selected[1:], s_alg_values, strict=True)
    )
    if any(not math.isfinite(point["delta_e"]) or point["delta_e"] < 0 for point in result):
        raise PageUpdateError(f"{label} contains an invalid same-cutoff error")
    return result


def point_at_round(points: Sequence[Mapping[str, Any]], round_index: int) -> Mapping[str, Any]:
    matches = [point for point in points if int(point["round"]) == int(round_index)]
    if len(matches) != 1:
        raise PageUpdateError(f"curve does not contain unique round {round_index}")
    return matches[0]


def make_curves(adapter: Mapping[str, Any]) -> dict[str, Any]:
    cells_by_regime = {str(cell["regime_id"]): cell for cell in adapter["cells"]}
    if tuple(cells_by_regime) != REGIME_ORDER:
        raise PageUpdateError("page-7 adapter regime order drifted")

    output_cells: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        cell = cells_by_regime[regime]
        append = cell["append"]
        append_source = append["source"]["archive"]
        append_archive = resolve_source_path(str(append_source["path"]))
        validate_archive_size(append_archive, append_source)
        append_s_alg = extract_s_alg(
            append_archive,
            member_name=APPEND_MEMBER,
            prefix=APPEND_PREFIX,
        )
        append_display_prefixes = len(append["points"]) - 1
        if len(append_s_alg) < append_display_prefixes:
            raise PageUpdateError(
                f"{regime} Append-ADAPT archive has fewer costs than displayed points"
            )
        append_s_alg = append_s_alg[:append_display_prefixes]
        append_curve = curve_from_points(
            append,
            append_s_alg,
            expected_terminal_s_alg=int(append["terminal"]["costs"]["S_alg"]),
            expected_s_alg_round=PAPER_FACING_COST_ROUND,
            label=f"{regime} Append-ADAPT",
        )

        ra = cell["ra"]
        if cell["status"] == "complete":
            ra_archive = locate_ra_complete_archive(cell)
            ra_source = ra["source"]["archive"]
            ra_member = RA_COMPLETE_MEMBER
            expected_ra_s_alg = int(ra["terminal"]["costs"]["S_alg"])
        else:
            ra_source = ra["source"]["snapshot_archive"]
            ra_archive = resolve_source_path(str(ra_source["path"]))
            validate_archive_size(ra_archive, ra_source)
            ra_member = RA_LIVE_MEMBER
            expected_ra_s_alg = int(ra["authenticated_cost_prefix"]["costs"]["S_alg"])
        ra_s_alg = extract_s_alg(ra_archive, member_name=ra_member, prefix=RA_PREFIX)
        ra_curve = curve_from_points(
            ra,
            ra_s_alg,
            expected_terminal_s_alg=expected_ra_s_alg,
            label=f"{regime} RA",
        )

        append_marker = point_at_round(
            append_curve, PAPER_FACING_COST_ROUND
        )
        if cell["status"] == "complete":
            ra_plateau_round = int(ra["effective_plateau"]["round"])
            ra_marker = point_at_round(ra_curve, ra_plateau_round)
            ra_marker_policy = "first_effective_plateau_prefix"
        else:
            available = ra.get("available_prefix_effective_plateau")
            available_round = int(available["round"]) if isinstance(available, Mapping) else -1
            if 0 <= available_round <= int(ra_curve[-1]["round"]):
                ra_marker = point_at_round(ra_curve, available_round)
                ra_marker_policy = "first_effective_plateau_prefix"
            else:
                ra_marker = ra_curve[-1]
                ra_marker_policy = "last_authenticated_plotted_prefix"

        output_cells.append(
            {
                "regime_id": regime,
                "display_name": str(cell["display_name"]).replace("--", "-"),
                "status": str(cell["status"]),
                "append": {
                    "execution_id": append["execution_id"],
                    "points": append_curve,
                    "marker": dict(append_marker),
                    "marker_policy": "paper_facing_cost_anchor_round_50",
                    "trajectory_terminal": dict(append_curve[-1]),
                    "source": {
                        "archive_path": str(append_archive),
                        "archive_expected_sha256": append_source["sha256"],
                        "archive_size_bytes": append_archive.stat().st_size,
                        "member": APPEND_MEMBER,
                        "json_prefix": APPEND_PREFIX,
                    },
                },
                "ra": {
                    "execution_id": ra["execution_id"],
                    "points": ra_curve,
                    "marker": dict(ra_marker),
                    "marker_policy": ra_marker_policy,
                    "source": {
                        "archive_path": str(ra_archive),
                        "archive_expected_sha256": ra_source["sha256"],
                        "archive_size_bytes": ra_archive.stat().st_size,
                        "member": ra_member,
                        "json_prefix": RA_PREFIX,
                    },
                    "unplotted_stdout_tail": (
                        None
                        if cell["status"] == "complete"
                        else {
                            "last_authenticated_round": int(ra_curve[-1]["round"]),
                            "latest_energy_only_round": int(ra["terminal"]["round"]),
                            "reason": "no checkpoint-backed estimator ledger after authenticated prefix",
                        }
                    ),
                },
            }
        )

    payload: dict[str, Any] = {
        "schema": "paper_i_ra_append_page7_deltae_vs_salg_curves_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": {
            "page": 7,
            "regimes": list(REGIME_ORDER),
            "methods": ["Append-ADAPT", "Global-singleton RA"],
            "displayed_error": "same-cutoff absolute energy error",
            "x_metric": "cumulative S_alg",
            "initial_state_policy": "round 0 plotted at S_alg=0 before adaptive work",
            "live_tail_policy": "stop at last authenticated estimator-ledger prefix; never infer stdout-only cost",
        },
        "source_adapter": binding(SOURCE_ADAPTER),
        "cells": output_cells,
    }
    payload["sha256"] = canonical_digest(payload)
    return payload


def render_page(curves: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import EngFormatter, LogLocator, MaxNLocator, NullFormatter

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    fig.text(
        0.5,
        0.965,
        r"Global-singleton RA versus fresh Append-ADAPT: $|\Delta E|$ versus $S_{\rm alg}$",
        ha="center",
        va="top",
        fontsize=16.0,
        weight="bold",
    )
    fig.text(
        0.5,
        0.929,
        r"Same-cutoff error against authenticated cumulative estimator work; no cost is inferred for stdout-only RA tails.",
        ha="center",
        va="top",
        fontsize=9.5,
    )
    grid = fig.add_gridspec(
        2,
        3,
        left=0.070,
        right=0.985,
        bottom=0.445,
        top=0.880,
        wspace=0.22,
        hspace=0.34,
    )
    cells = {str(cell["regime_id"]): cell for cell in curves["cells"]}
    for index, regime in enumerate(REGIME_ORDER):
        ax = fig.add_subplot(grid[index // 3, index % 3])
        cell = cells[regime]
        for method_key, color in (("append", APPEND_COLOR), ("ra", RA_COLOR)):
            method = cell[method_key]
            x = [int(point["S_alg"]) for point in method["points"]]
            y = [max(float(point["delta_e"]), PLOT_FLOOR) for point in method["points"]]
            ax.plot(x, y, color=color, linewidth=1.8)
            marker = method["marker"]
            ax.scatter(
                [int(marker["S_alg"])],
                [max(float(marker["delta_e"]), PLOT_FLOOR)],
                marker="o" if method_key == "append" else "D",
                color=color,
                s=31,
                zorder=5,
            )
        if cell["ra"]["marker_policy"] == "last_authenticated_plotted_prefix":
            marker = cell["ra"]["marker"]
            ax.annotate(
                f"RA auth. k={int(marker['round'])}",
                (int(marker["S_alg"]), max(float(marker["delta_e"]), PLOT_FLOOR)),
                xytext=(-4, 7),
                textcoords="offset points",
                ha="right",
                fontsize=6.7,
                color="#7A2E2E",
            )
        ax.set_title(str(cell["display_name"]), pad=4)
        ax.set_yscale("log")
        xmax = max(
            int(cell["append"]["points"][-1]["S_alg"]),
            int(cell["ra"]["points"][-1]["S_alg"]),
        )
        ax.set_xlim(0, xmax * 1.035)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.xaxis.set_major_formatter(EngFormatter(sep=""))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=70)
        )
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="major", color="#d8d8d8", linewidth=0.55)
        ax.grid(which="minor", color="#efefef", linewidth=0.35)
        if index % 3 == 0:
            ax.set_ylabel(r"Same-cutoff $|\Delta E|$")
        if index >= 3:
            ax.set_xlabel(r"Cumulative $S_{\rm alg}$")

    fig.legend(
        handles=[
            Line2D([0], [0], color=APPEND_COLOR, marker="o", label="Append-ADAPT"),
            Line2D([0], [0], color=RA_COLOR, marker="D", label="Global-singleton RA"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.385),
        ncol=2,
        frameon=False,
        fontsize=8.7,
        title=(
            "Append marker/table: authenticated k=50 cost anchor; "
            "RA marker: first plateau or last authenticated prefix"
        ),
        title_fontsize=7.5,
    )

    table_ax = fig.add_axes([0.055, 0.130, 0.89, 0.185])
    table_ax.axis("off")
    columns = [
        "Regime",
        "RA plotted k",
        r"RA $S_{\rm alg}$",
        r"RA $|\Delta E|$",
        "Append k",
        r"Append $S_{\rm alg}$",
        r"Append $|\Delta E|$",
    ]
    rows: list[list[str]] = []
    for regime in REGIME_ORDER:
        cell = cells[regime]
        ra_terminal = cell["ra"]["points"][-1]
        append_terminal = point_at_round(
            cell["append"]["points"], PAPER_FACING_COST_ROUND
        )
        rows.append(
            [
                str(cell["display_name"]),
                str(int(ra_terminal["round"])),
                f"{int(ra_terminal['S_alg']):,}",
                f"{float(ra_terminal['delta_e']):.2e}",
                str(int(append_terminal["round"])),
                f"{int(append_terminal['S_alg']):,}",
                f"{float(append_terminal['delta_e']):.2e}",
            ]
        )
    table = table_ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.18, 0.11, 0.13, 0.13, 0.10, 0.14, 0.14],
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#cfd4dc")
        cell.set_linewidth(0.5)
        if row_index == 0:
            cell.set_facecolor("#e8ebf0")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("white" if row_index % 2 else "#f7f7f7")
            if col_index == 0:
                cell.get_text().set_ha("left")

    fig.text(
        0.055,
        0.098,
        r"$S_{\rm alg}=N_{H,{\rm outer}}+N_{H,{\rm refit}}+N_{\rm grad}+N_{\rm metric}$ from cumulative executed-query receipts. "
        "The three strong-phonon RA curves stop at authenticated k=49, 45, and 31;\n"
        "later energy-only stdout points are excluded because no cumulative estimator ledger is available.",
        ha="left",
        va="top",
        fontsize=7.1,
    )
    fig.text(
        0.055,
        0.045,
        "Evolving diagnostic only; not adopted Paper-I evidence. Initial reference state is shown at k=0, S_alg=0.",
        ha="left",
        va="top",
        fontsize=6.8,
        color="#555555",
    )
    fig.savefig(PAGE_PDF)
    fig.savefig(PAGE_PNG, dpi=220)
    plt.close(fig)


def page_content_hash(page: Any) -> str:
    contents = page.get_contents()
    data = b"" if contents is None else contents.get_data()
    return hashlib.sha256(data).hexdigest()


def replace_page(curves: Mapping[str, Any]) -> dict[str, Any]:
    from pypdf import PdfReader, PdfWriter

    provenance = load_object(MASTER_PROVENANCE)
    expected_pdf = provenance.get("outputs", {}).get("partial_progress_pdf", {})
    if not isinstance(expected_pdf, Mapping) or expected_pdf.get("sha256") != sha256_file(MASTER_PDF):
        raise PageUpdateError("current report is not bound by its provenance")
    reader = PdfReader(str(MASTER_PDF), strict=False)
    layout = provenance.get("layout")
    if not isinstance(layout, Mapping):
        raise PageUpdateError("current report has no layout contract")
    try:
        page_count = int(layout.get("page_count", -1))
    except (TypeError, ValueError) as exc:
        raise PageUpdateError("current report page count is invalid") from exc
    if page_count < 7 or len(reader.pages) != page_count:
        raise PageUpdateError(
            "page-7 replacement requires a provenance-bound report with at least seven pages"
        )
    page_reader = PdfReader(str(PAGE_PDF), strict=False)
    if len(page_reader.pages) != 1:
        raise PageUpdateError("replacement page asset is not one page")
    replacement = page_reader.pages[0]
    if (float(replacement.mediabox.width), float(replacement.mediabox.height)) != (792.0, 612.0):
        raise PageUpdateError("replacement page is not landscape letter")

    before_hashes = [page_content_hash(page) for page in reader.pages]
    writer = PdfWriter()
    for index, page in enumerate(reader.pages):
        writer.add_page(replacement if index == 6 else page)
    if reader.metadata:
        writer.add_metadata(dict(reader.metadata))

    temporary_pdf = MASTER_PDF.with_suffix(".page7-salg.tmp.pdf")
    temporary_provenance = MASTER_PROVENANCE.with_suffix(".page7-salg.tmp.json")
    rollback_pdf = MASTER_PDF.with_suffix(".page7-salg.rollback.pdf")
    for path in (temporary_pdf, temporary_provenance, rollback_pdf):
        path.unlink(missing_ok=True)
    with temporary_pdf.open("xb") as stream:
        writer.write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    updated_reader = PdfReader(str(temporary_pdf), strict=False)
    after_hashes = [page_content_hash(page) for page in updated_reader.pages]
    if len(updated_reader.pages) != page_count:
        raise PageUpdateError("updated report page count drifted")
    if after_hashes[:6] != before_hashes[:6] or after_hashes[7:] != before_hashes[7:]:
        raise PageUpdateError("page-7 replacement changed a preserved page")

    updated = copy.deepcopy(provenance)
    updated["layout"]["page_7"] = "historical_mean_global_singleton_deltae_vs_salg_v1"
    completion = updated.get("historical_mean_global_singleton_cost_completion")
    if isinstance(completion, dict):
        completion["presentation"] = (
            "Page 7 plots same-cutoff energy error against authenticated cumulative S_alg."
        )
        validation = completion.get("structural_validation")
        if isinstance(validation, dict):
            validation["new_page_content_sha256"] = after_hashes[5:7]
        outputs = completion.get("outputs")
        if isinstance(outputs, dict):
            outputs["accuracy_page_pdf"] = binding(PAGE_PDF)
            outputs["accuracy_plot_png"] = binding(PAGE_PNG)

    updated["page_7_deltae_vs_salg"] = {
        "schema": "paper_i_ra_append_page7_deltae_vs_salg_update_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "curve_source": binding(CURVE_JSON),
        "page_asset_pdf": binding(PAGE_PDF),
        "page_asset_png": binding(PAGE_PNG),
        "plot_contract": {
            "x": "authenticated cumulative S_alg",
            "y": "same-cutoff absolute energy error",
            "y_scale": "log",
            "x_scale": "linear",
            "line_markers": (
                "Append marker is the authenticated round-50 cost anchor; "
                "RA marker is the first plateau or last authenticated prefix"
            ),
            "stdout_only_ra_tail": "not plotted",
        },
        "structural_validation": {
            "page_count": page_count,
            "replaced_page": 7,
            "preserved_page_content_sha256": before_hashes[:6] + before_hashes[7:],
            "replacement_page_content_sha256": after_hashes[6],
        },
        "curve_summary": [
            {
                "regime_id": cell["regime_id"],
                "append_point_count": len(cell["append"]["points"]),
                "append_marker": cell["append"]["marker"],
                "append_cost_anchor_round_50": point_at_round(
                    cell["append"]["points"], PAPER_FACING_COST_ROUND
                ),
                "append_trajectory_terminal_round_70": cell["append"][
                    "points"
                ][-1],
                "ra_point_count": len(cell["ra"]["points"]),
                "ra_marker": cell["ra"]["marker"],
                "ra_marker_policy": cell["ra"]["marker_policy"],
                "ra_terminal": cell["ra"]["points"][-1],
                "ra_unplotted_stdout_tail": cell["ra"]["unplotted_stdout_tail"],
            }
            for cell in curves["cells"]
        ],
    }
    updated["limitations"] = [
        limitation
        for limitation in updated.get("limitations", [])
        if not str(limitation).startswith("Page 7 places the accuracy panels")
        and not str(limitation).startswith("Page 7 plots only checkpoint-authenticated")
    ]
    updated["limitations"].append(
        "Page 7 plots only checkpoint-authenticated cumulative S_alg. The weak-strong, "
        "intermediate-strong, and strong-strong RA curves stop at k=49, k=45, and k=31; "
        "their later stdout-only energy points are not assigned inferred costs."
    )
    updated["outputs"]["partial_progress_pdf"] = {
        **binding(temporary_pdf),
        "path": str(MASTER_PDF.resolve()),
    }

    try:
        with temporary_provenance.open("xb") as stream:
            stream.write(
                json.dumps(updated, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
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
        "status": "page_7_replaced_with_deltae_vs_salg",
        "page_count": page_count,
        "pdf_sha256": sha256_file(MASTER_PDF),
        "curve_json": str(CURVE_JSON),
        "page_pdf": str(PAGE_PDF),
        "page_png": str(PAGE_PNG),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--curves-only", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        adapter = merge_append_r70(
            load_object(SOURCE_ADAPTER), load_object(APPEND_R70_ADAPTER)
        )
        curves = make_curves(adapter)
        CURVE_JSON.write_text(
            json.dumps(curves, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        render_page(curves)
        result = (
            {
                "status": "curves_and_page_asset_built",
                "curve_json": str(CURVE_JSON),
                "page_pdf": str(PAGE_PDF),
                "page_png": str(PAGE_PNG),
            }
            if args.curves_only
            else replace_page(curves)
        )
    except (OSError, PageUpdateError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Refresh Paper-I HH error-vs-iteration continuation plots.

This script is intentionally narrow. It stitches completed SNAKE continuation
artifacts onto the existing physical-lane traces for the continuation regimes,
keeps the current Geo-ADAPT and Append-ADAPT comparator curves, and updates only
the plot/provenance artifacts that reference those convergence panels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "MATH/paper_details/figures/paper_i_physical_lane_snake_duplicate_20260708"
STEM = "paper_i_physical_lane_snake_duplicate_20260708"
PLOT_PNG = FIG_DIR / f"{STEM}__weak_strong.png"
PLOT_PDF = FIG_DIR / f"{STEM}__weak_strong.pdf"
INTERMEDIATE_PLOT_PNG = FIG_DIR / f"{STEM}__intermediate_strong.png"
INTERMEDIATE_PLOT_PDF = FIG_DIR / f"{STEM}__intermediate_strong.pdf"
PLOT_PROV_JSON = FIG_DIR / f"{STEM}_append_parent_only_provenance.json"
PLOT_PROV_CSV = FIG_DIR / f"{STEM}_append_parent_only_provenance.csv"
STITCHED_JSON = FIG_DIR / f"{STEM}__weak_strong_snake_depth50_stitched_source.json"
INTERMEDIATE_STITCHED_JSON = FIG_DIR / f"{STEM}__intermediate_strong_snake_depth45_stitched_source.json"
PAPER_I_PROVENANCE_DIR = (
    REPO_ROOT / "MATH/paper_facing/paper_I_static_scaffold/provenance"
)
ROOT_PROV_JSON = PAPER_I_PROVENANCE_DIR / "Paper_I_provenance.json"
ROOT_PROV_CSV = PAPER_I_PROVENANCE_DIR / "Paper_I_provenance.csv"
ROOT_PROV_TXT = PAPER_I_PROVENANCE_DIR / "Paper_I_provenance.txt"
PAPER_TEX = REPO_ROOT / "MATH/paper_details/Paper_I.tex"
COMMENT_BEGIN = "% BEGIN_MACHINE_READABLE_HH_PHYSICAL_LANE_DUPLICATE_UPDATE_20260708"
COMMENT_END = "% END_MACHINE_READABLE_HH_PHYSICAL_LANE_DUPLICATE_UPDATE_20260708"

DEPTH30 = REPO_ROOT / "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/weak_strong/json/result.json"
DEPTH45 = REPO_ROOT / "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_depth45_continuation_20260708/weak_strong/json/result.json"
FETCH_GLOB = "raw_outputs/chtc_fetches/paper_i_hh_weak_strong_depth50_20260708_v1_proc0_snake_fetch_*/raw_outputs/paper_i_hh_weak_strong_depth50_20260708_v1/paper_i_hh_weak_strong_depth50_20260708_v1__weak_strong__snake__physical_operator_lane_x3_nobatch__powell200__depth50_continuation_from45/json/result.json"
INTERMEDIATE_DEPTH30 = REPO_ROOT / "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/intermediate_strong/json/result.json"
INTERMEDIATE_DEPTH45 = REPO_ROOT / "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_depth45_continuation_20260708/intermediate_strong/json/result.json"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def as_int(value: Any) -> int | None:
    out = as_float(value)
    if out is None:
        return None
    return int(round(out))


def result_error(payload: dict[str, Any]) -> float:
    adapt = payload.get("adapt_vqe", {})
    if not isinstance(adapt, dict):
        raise ValueError("missing adapt_vqe payload")
    energy = as_float(adapt.get("energy"))
    exact = as_float(adapt.get("exact_gs_energy"))
    if energy is not None and exact is not None:
        return abs(energy - exact)
    direct = as_float(adapt.get("abs_delta_e") or adapt.get("benchmark_target_abs_delta_e_current"))
    if direct is None:
        raise ValueError("missing terminal energy error")
    return abs(direct)


def final_refit_error(payload: dict[str, Any]) -> float | None:
    adapt = payload.get("adapt_vqe", {})
    if not isinstance(adapt, dict):
        return None
    exact = as_float(adapt.get("exact_gs_energy"))
    refit = adapt.get("final_full_refit")
    if not isinstance(refit, dict) or exact is None:
        return None
    energy = as_float(refit.get("energy_after"))
    return abs(energy - exact) if energy is not None else None


def history_error(row: dict[str, Any]) -> float | None:
    for key in (
        "delta_abs_current",
        "benchmark_target_abs_delta_current",
        "abs_delta_e_after",
        "abs_delta_e_same_cutoff_after",
        "exact_abs_delta_e_from_final_state",
        "abs_delta_e",
    ):
        out = as_float(row.get(key))
        if out is not None and out > 0.0:
            return out
    return None


def stitch_segment(path: Path, include_zero: bool) -> tuple[dict[int, float], dict[str, Any]]:
    payload = read_json(path)
    adapt = payload.get("adapt_vqe", {})
    history = adapt.get("history") if isinstance(adapt, dict) else None
    if not isinstance(history, list) or not history:
        raise ValueError(f"missing ADAPT history in {path}")

    points: dict[int, float] = {}
    if include_zero:
        initial = history_error(history[0])
        if initial is not None:
            points[0] = initial

    for idx, row in enumerate(history, start=1):
        if not isinstance(row, dict):
            continue
        x = as_int(row.get("depth_cumulative"))
        if x is None:
            x = as_int(row.get("depth")) or idx
        y = history_error(row)
        if y is not None:
            points[int(x)] = y

    terminal_x = max(points) if points else len(history)
    settings = payload.get("settings", {})
    if isinstance(settings, dict):
        terminal_x = as_int(settings.get("adapt_segment_target_depth")) or terminal_x
    # For continuation plots the x-axis is the cumulative adaptive-step budget,
    # not the post-prune accepted operator count. Keep the depth-50 terminal
    # result at k=50 even when pruning leaves ansatz_depth/final_depth at 49.
    points[int(terminal_x)] = result_error(payload)

    return points, {
        "path": rel(path),
        "sha256": sha256(path),
        "history_len": len(history),
        "terminal_x": int(terminal_x),
        "terminal_abs_delta_e": result_error(payload),
        "final_refit_abs_delta_e": final_refit_error(payload),
    }


def snake_depth50_result() -> Path:
    matches = sorted(REPO_ROOT.glob(FETCH_GLOB))
    if not matches:
        raise FileNotFoundError(f"No fetched depth-50 result matched {FETCH_GLOB}")
    return matches[-1]


def stitched_snake_points() -> tuple[list[tuple[int, float]], dict[str, Any]]:
    merged: dict[int, float] = {}
    sources = []
    for path, include_zero in ((DEPTH30, True), (DEPTH45, False), (snake_depth50_result(), False)):
        points, summary = stitch_segment(path, include_zero=include_zero)
        merged.update(points)
        sources.append(summary)
    series = sorted(merged.items())
    return series, {
        "sources": sources,
        "terminal_k": 50,
        "terminal_abs_delta_e": merged[50],
        "point_count": len(series),
    }


def stitched_intermediate_strong_points() -> tuple[list[tuple[int, float]], dict[str, Any]]:
    merged: dict[int, float] = {}
    sources = []
    for path, include_zero in ((INTERMEDIATE_DEPTH30, True), (INTERMEDIATE_DEPTH45, False)):
        points, summary = stitch_segment(path, include_zero=include_zero)
        merged.update(points)
        sources.append(summary)
    series = sorted(merged.items())
    return series, {
        "sources": sources,
        "terminal_k": 45,
        "terminal_abs_delta_e": merged[45],
        "point_count": len(series),
    }


def comparator_rows(regime: str) -> dict[str, dict[str, Any]]:
    plot_prov = read_json(PLOT_PROV_JSON)
    support_csv = Path(plot_prov["support_csv"])
    if not support_csv.exists():
        raise FileNotFoundError(support_csv)
    rows: dict[str, dict[str, Any]] = {}
    with support_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("regime") != regime:
                continue
            role = row.get("role_key")
            if role in {"geo_macro_c", "append_macro_c"}:
                rows[role] = row
    missing = {"geo_macro_c", "append_macro_c"} - set(rows)
    if missing:
        raise ValueError(f"Missing comparator support rows: {sorted(missing)}")
    return rows


def parse_points(raw: str) -> list[tuple[int, float]]:
    data = json.loads(raw)
    points: list[tuple[int, float]] = []
    for item in data:
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        x = as_int(item[0])
        y = as_float(item[1])
        if x is not None and y is not None and y > 0.0:
            points.append((x, y))
    return points


def y_at(points: list[tuple[int, float]], k: int) -> float:
    for x, y in points:
        if int(x) == int(k):
            return y
    raise ValueError(f"missing point k={k}")


def render_plot(
    snake_points: list[tuple[int, float]],
    geo_points: list[tuple[int, float]],
    append_points: list[tuple[int, float]],
    *,
    plot_png: Path,
    plot_pdf: Path,
    title: str,
    x_right: int,
    marker_ks: dict[str, int],
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.35), dpi=220)
    styles = {
        "SNAKE": ("#E45756", 3.0, "*"),
        "Geo-ADAPT": ("#54A24B", 2.1, "^"),
        "Append-ADAPT": ("#4C78A8", 2.1, "o"),
    }
    for label, points in (
        ("SNAKE", snake_points),
        ("Geo-ADAPT", geo_points),
        ("Append-ADAPT", append_points),
    ):
        color, width, _marker = styles[label]
        ax.plot([x for x, _ in points], [y for _, y in points], color=color, linewidth=width, linestyle="-")

    point_map = {
        "SNAKE": snake_points,
        "Geo-ADAPT": geo_points,
        "Append-ADAPT": append_points,
    }
    markers = {
        label: (k, y_at(point_map[label], k))
        for label, k in marker_ks.items()
    }
    for label, (x, y) in markers.items():
        color, _width, marker = styles[label]
        size = 118 if label == "SNAKE" else 62
        ax.scatter([x], [y], marker=marker, s=size, color=color, edgecolors="black", linewidths=0.7, zorder=8)

    all_y = [y for _, y in snake_points + geo_points + append_points if y > 0.0]
    ax.set_yscale("log")
    ax.set_ylim(max(1e-8, min(all_y) / 2.5), max(all_y) * 1.8)
    ax.set_xlim(left=0, right=x_right)
    ax.set_xlabel(r"ADAPT iteration $k$")
    ax.set_ylabel(r"$|\Delta E|$")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=80))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linestyle="-", linewidth=0.55, alpha=0.22)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.13)

    handles = []
    for label, (k, _y) in markers.items():
        color, width, marker = styles[label]
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="-",
                linewidth=width,
                marker=marker,
                markersize=8 if label == "SNAKE" else 6,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label=f"{label}, k={k}",
            )
        )
    ax.legend(handles=handles, frameon=True, framealpha=0.94, fontsize=8.0, loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_png, bbox_inches="tight")
    fig.savefig(plot_pdf, bbox_inches="tight")
    plt.close(fig)


def update_plot_provenance(weak_summary: dict[str, Any], intermediate_summary: dict[str, Any]) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    plot_prov = read_json(PLOT_PROV_JSON)
    plot_prov["generated_utc"] = now
    plot_prov["depth50_update"] = {
        "regime": "weak-strong",
        "method": "SNAKE",
        "source": rel(STITCHED_JSON),
        "source_sha256": sha256(STITCHED_JSON),
        "terminal_k": weak_summary["terminal_k"],
        "terminal_abs_delta_e": weak_summary["terminal_abs_delta_e"],
        "policy": "SNAKE curve stitched from depth-30 source, depth-45 continuation, and completed CHTC depth-50 continuation; Geo/Append curves remain current 30-iteration parent-pool comparator sources.",
    }
    plot_prov["intermediate_strong_depth45_update"] = {
        "regime": "intermediate-strong",
        "method": "SNAKE",
        "source": rel(INTERMEDIATE_STITCHED_JSON),
        "source_sha256": sha256(INTERMEDIATE_STITCHED_JSON),
        "terminal_k": intermediate_summary["terminal_k"],
        "terminal_abs_delta_e": intermediate_summary["terminal_abs_delta_e"],
        "policy": "SNAKE curve stitched from depth-30 source and completed local depth-45 continuation; Geo/Append curves remain current 30-iteration parent-pool comparator sources.",
    }
    for plot in plot_prov.get("plots", []):
        regime = plot.get("regime")
        if regime not in {"weak-strong", "intermediate-strong"}:
            continue
        if regime == "weak-strong":
            plot_png = PLOT_PNG
            plot_pdf = PLOT_PDF
            summary = weak_summary
            stitched_json = STITCHED_JSON
            marker_source = "terminal_depth50_result"
            initial_policy = "stitched depth-30/depth-45/depth-50 cumulative trajectory; x=0 uses the depth-30 first post-admission error; terminal segment uses the completed CHTC depth-50 result energy"
        else:
            plot_png = INTERMEDIATE_PLOT_PNG
            plot_pdf = INTERMEDIATE_PLOT_PDF
            summary = intermediate_summary
            stitched_json = INTERMEDIATE_STITCHED_JSON
            marker_source = "terminal_depth45_result"
            initial_policy = "stitched depth-30/depth-45 cumulative trajectory; x=0 uses the depth-30 first post-admission error; terminal segment uses the completed local depth-45 final-refit energy"
        plot["png_sha256"] = sha256(plot_png)
        plot["pdf_sha256"] = sha256(plot_pdf)
        for method in plot.get("methods", []):
            if method.get("method") != "SNAKE":
                continue
            method.update(
                {
                    "point_count": summary["point_count"],
                    "marker_k": summary["terminal_k"],
                    "marker_error": summary["terminal_abs_delta_e"],
                    "marker_source": marker_source,
                    "source_json": rel(stitched_json),
                    "source_sha256": sha256(stitched_json),
                    "table_error": summary["terminal_abs_delta_e"],
                    "initial_point_policy": initial_policy,
                }
            )
    write_json(PLOT_PROV_JSON, plot_prov)

    rows: list[dict[str, Any]] = []
    for plot in plot_prov.get("plots", []):
        png = plot["png"]
        pdf = plot["pdf"]
        for method in plot.get("methods", []):
            rows.append(
                {
                    "regime": plot["regime"],
                    "method": method["method"],
                    "role_key": method["role_key"],
                    "point_count": method["point_count"],
                    "marker_k": method["marker_k"],
                    "marker_error": method["marker_error"],
                    "table_error": method["table_error"],
                    "marker_source": method["marker_source"],
                    "initial_point_policy": method["initial_point_policy"],
                    "png": png,
                    "pdf": pdf,
                    "source_json": method["source_json"],
                    "source_sha256": method["source_sha256"],
                }
            )
    with PLOT_PROV_CSV.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "regime",
            "method",
            "role_key",
            "point_count",
            "marker_k",
            "marker_error",
            "table_error",
            "marker_source",
            "initial_point_policy",
            "png",
            "pdf",
            "source_json",
            "source_sha256",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_root_provenance(weak_summary: dict[str, Any], intermediate_summary: dict[str, Any]) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    root = read_json(ROOT_PROV_JSON)
    root["generated_utc"] = now
    root["append_parent_only_plot_provenance_json_sha256"] = sha256(PLOT_PROV_JSON)
    root["append_parent_only_plot_provenance_csv_sha256"] = sha256(PLOT_PROV_CSV)
    root["continuation_status"] = "weak-strong SNAKE depth-50 continuation and intermediate-strong SNAKE depth-45 continuation are used for their error-vs-iteration plots; Geo/Append comparator curves remain current 30-iteration parent-pool sources."
    root["depth50_plot_update"] = {
        "regime": "weak-strong",
        "method": "SNAKE",
        "stitched_source_json": rel(STITCHED_JSON),
        "stitched_source_json_sha256": sha256(STITCHED_JSON),
        "terminal_k": weak_summary["terminal_k"],
        "terminal_abs_delta_e": weak_summary["terminal_abs_delta_e"],
        "plot_png": rel(PLOT_PNG),
        "plot_png_sha256": sha256(PLOT_PNG),
        "plot_pdf": rel(PLOT_PDF),
        "plot_pdf_sha256": sha256(PLOT_PDF),
    }
    root["intermediate_strong_depth45_plot_update"] = {
        "regime": "intermediate-strong",
        "method": "SNAKE",
        "stitched_source_json": rel(INTERMEDIATE_STITCHED_JSON),
        "stitched_source_json_sha256": sha256(INTERMEDIATE_STITCHED_JSON),
        "terminal_k": intermediate_summary["terminal_k"],
        "terminal_abs_delta_e": intermediate_summary["terminal_abs_delta_e"],
        "plot_png": rel(INTERMEDIATE_PLOT_PNG),
        "plot_png_sha256": sha256(INTERMEDIATE_PLOT_PNG),
        "plot_pdf": rel(INTERMEDIATE_PLOT_PDF),
        "plot_pdf_sha256": sha256(INTERMEDIATE_PLOT_PDF),
    }
    weak_plot = root.setdefault("plots", {}).setdefault("weak_strong", {})
    weak_plot.update(
        {
            "png_sha256": sha256(PLOT_PNG),
            "pdf_sha256": sha256(PLOT_PDF),
            "snake_marker_k": weak_summary["terminal_k"],
            "snake_marker_error": weak_summary["terminal_abs_delta_e"],
            "snake_point_count": weak_summary["point_count"],
            "snake_stitched_source_json": rel(STITCHED_JSON),
            "snake_stitched_source_json_sha256": sha256(STITCHED_JSON),
        }
    )
    intermediate_plot = root.setdefault("plots", {}).setdefault("intermediate_strong", {})
    intermediate_plot.update(
        {
            "png_sha256": sha256(INTERMEDIATE_PLOT_PNG),
            "pdf_sha256": sha256(INTERMEDIATE_PLOT_PDF),
            "snake_marker_k": intermediate_summary["terminal_k"],
            "snake_marker_error": intermediate_summary["terminal_abs_delta_e"],
            "snake_point_count": intermediate_summary["point_count"],
            "snake_stitched_source_json": rel(INTERMEDIATE_STITCHED_JSON),
            "snake_stitched_source_json_sha256": sha256(INTERMEDIATE_STITCHED_JSON),
        }
    )
    write_json(ROOT_PROV_JSON, root)

    with ROOT_PROV_CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = rows[0].keys() if rows else []
    for row in rows:
        if row.get("regime") == "weak-strong":
            row["plot_png_sha256"] = sha256(PLOT_PNG)
        if row.get("regime") == "intermediate-strong":
            row["plot_png_sha256"] = sha256(INTERMEDIATE_PLOT_PNG)
    with ROOT_PROV_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "Paper_I duplicate provenance",
        f"generated_utc: {now}",
        f"source_pdf: {root.get('source_pdf')}",
        f"duplicate_tex: {root.get('duplicate_tex')}",
        f"duplicate_pdf_local: {root.get('duplicate_pdf_local')}",
        f"final_pdf: {root.get('final_pdf')}",
        f"comparison_pdf: {root.get('comparison_pdf')}",
        f"comparison_csv: {root.get('comparison_csv')}",
        f"append_parent_only_plot_provenance_json: {root.get('append_parent_only_plot_provenance_json')}",
        "policy: visible HH tables display existing SNAKE, Geo-ADAPT, and Append-ADAPT rows; the weak-strong plot extends SNAKE through the completed depth-50 continuation and the intermediate-strong plot extends SNAKE through the completed depth-45 continuation.",
        "continuation_note: weak-strong depth-50 and intermediate-strong depth-45 SNAKE continuations are stitched into their plots; table/cost cells are intentionally unchanged in this plot-only refresh.",
        "visible_label_update: table and plot labels use SNAKE, Geo-ADAPT, and Append-ADAPT; parent/singleton labels are not rendered.",
    ]
    ROOT_PROV_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_tex_comment() -> None:
    root = read_json(ROOT_PROV_JSON)
    text = PAPER_TEX.read_text(encoding="utf-8")
    begin = text.index(COMMENT_BEGIN)
    end = text.index(COMMENT_END, begin)
    payload = json.dumps(root, indent=2, sort_keys=True)
    commented = "\n".join("% " + line if line else "%" for line in payload.splitlines())
    replacement = f"{COMMENT_BEGIN}\n{commented}\n{COMMENT_END}"
    text = text[:begin] + replacement + text[end + len(COMMENT_END) :]
    PAPER_TEX.write_text(text, encoding="utf-8")


def main() -> None:
    snake_points, stitched_summary = stitched_snake_points()
    comparators = comparator_rows("weak-strong")
    geo_points = parse_points(comparators["geo_macro_c"]["trajectory_points_json"])
    append_points = parse_points(comparators["append_macro_c"]["trajectory_points_json"])

    render_plot(
        snake_points,
        geo_points,
        append_points,
        plot_png=PLOT_PNG,
        plot_pdf=PLOT_PDF,
        title=r"weak-strong: $U/t = 0.25$, $\lambda = 1.25$, $M = 4$",
        x_right=50,
        marker_ks={"SNAKE": 50, "Geo-ADAPT": 8, "Append-ADAPT": 23},
    )
    stitched_payload = {
        "schema": "paper_i_hh_weak_strong_depth50_stitched_curve_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "regime": "weak-strong",
        "method": "SNAKE",
        "point_policy": "stitched cumulative ADAPT-iteration curve from depth-30 source, depth-45 continuation, and completed CHTC depth-50 continuation",
        "points": [{"k": x, "abs_delta_e": y} for x, y in snake_points],
        **stitched_summary,
    }
    write_json(STITCHED_JSON, stitched_payload)

    intermediate_points, intermediate_summary = stitched_intermediate_strong_points()
    intermediate_comparators = comparator_rows("intermediate-strong")
    intermediate_geo_points = parse_points(intermediate_comparators["geo_macro_c"]["trajectory_points_json"])
    intermediate_append_points = parse_points(intermediate_comparators["append_macro_c"]["trajectory_points_json"])
    render_plot(
        intermediate_points,
        intermediate_geo_points,
        intermediate_append_points,
        plot_png=INTERMEDIATE_PLOT_PNG,
        plot_pdf=INTERMEDIATE_PLOT_PDF,
        title=r"intermediate-strong: $U/t = 1.25$, $\lambda = 1.25$, $M = 4$",
        x_right=45,
        marker_ks={"SNAKE": 45, "Geo-ADAPT": 8, "Append-ADAPT": 25},
    )
    intermediate_payload = {
        "schema": "paper_i_hh_intermediate_strong_depth45_stitched_curve_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "regime": "intermediate-strong",
        "method": "SNAKE",
        "point_policy": "stitched cumulative ADAPT-iteration curve from depth-30 source and completed local depth-45 continuation",
        "points": [{"k": x, "abs_delta_e": y} for x, y in intermediate_points],
        **intermediate_summary,
    }
    write_json(INTERMEDIATE_STITCHED_JSON, intermediate_payload)

    update_plot_provenance(stitched_summary, intermediate_summary)
    update_root_provenance(stitched_summary, intermediate_summary)
    update_tex_comment()

    print(json.dumps({
        "weak_strong": {
            "plot_png": rel(PLOT_PNG),
            "plot_png_sha256": sha256(PLOT_PNG),
            "plot_pdf": rel(PLOT_PDF),
            "plot_pdf_sha256": sha256(PLOT_PDF),
            "stitched_source_json": rel(STITCHED_JSON),
            "terminal_k": stitched_summary["terminal_k"],
            "terminal_abs_delta_e": stitched_summary["terminal_abs_delta_e"],
            "point_count": stitched_summary["point_count"],
        },
        "intermediate_strong": {
            "plot_png": rel(INTERMEDIATE_PLOT_PNG),
            "plot_png_sha256": sha256(INTERMEDIATE_PLOT_PNG),
            "plot_pdf": rel(INTERMEDIATE_PLOT_PDF),
            "plot_pdf_sha256": sha256(INTERMEDIATE_PLOT_PDF),
            "stitched_source_json": rel(INTERMEDIATE_STITCHED_JSON),
            "terminal_k": intermediate_summary["terminal_k"],
            "terminal_abs_delta_e": intermediate_summary["terminal_abs_delta_e"],
            "point_count": intermediate_summary["point_count"],
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

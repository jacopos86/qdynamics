#!/usr/bin/env python3
"""Build same-seed AP-McLachlan, AVQDS, and AVQDS(T) comparison plots."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_STYLES = {
    "AP-McLachlan": {"color": "#16845b", "linewidth": 1.65},
    "AVQDS": {"color": "#2855a6", "linewidth": 1.55},
    "AVQDS(T)": {"color": "#d46b08", "linewidth": 1.55},
}
EXACT_STYLE = {"color": "#171717", "linewidth": 1.15, "linestyle": "--"}
SEED_EXACT_STYLE = {"color": "#777777", "linewidth": 1.05, "linestyle": ":"}


@dataclass(frozen=True)
class ComparisonCase:
    key: str
    label: str
    ap_json: Path
    avqds_json: Path
    avqds_t_json: Path
    ap_cost_json: Path | None = None
    avqds_cost_json: Path | None = None
    avqds_t_cost_json: Path | None = None


@dataclass(frozen=True)
class MethodSeries:
    label: str
    source_path: Path
    time: np.ndarray
    energy: np.ndarray
    doublon: np.ndarray
    site_occupations: np.ndarray
    residual: np.ndarray
    append_times: tuple[float, ...]
    prune_times: tuple[float, ...]
    stabilization_spans: tuple[tuple[float, float], ...]
    stabilized_checkpoint_count: int


@dataclass(frozen=True)
class LoadedComparison:
    case: ComparisonCase
    methods: tuple[MethodSeries, ...]
    exact_energy: np.ndarray
    seed_exact_energy: np.ndarray
    exact_doublon: np.ndarray
    seed_exact_doublon: np.ndarray
    exact_site_occupations: np.ndarray
    seed_exact_site_occupations: np.ndarray
    terminal_costs: Mapping[str, Mapping[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Comparison input must be a JSON object: {path}")
    return dict(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(payload: Mapping[str, Any], *, source_path: Path) -> list[dict[str, Any]]:
    raw = payload.get("plot_rows")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"plot_rows must be a sequence: {source_path}")
    rows = [dict(row) for row in raw if isinstance(row, Mapping)]
    if len(rows) != len(raw) or not rows:
        raise ValueError(f"plot_rows contains invalid or no rows: {source_path}")
    return rows


def _numeric(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = np.nan
        values.append(number if np.isfinite(number) else np.nan)
    return np.asarray(values, dtype=float)


def _site_occupations(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> np.ndarray:
    values = np.full((len(rows), 2), np.nan, dtype=float)
    for index, row in enumerate(rows):
        try:
            occupation = np.asarray(row.get(key), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if occupation.size >= 2:
            values[index, :] = occupation[:2]
    return values


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _event_times(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    append_times: set[float] = set()
    prune_times: set[float] = set()
    for row in rows:
        if not bool(row.get("patch_accepted")):
            continue
        time = row.get("time")
        try:
            time_value = float(time)
        except (TypeError, ValueError):
            continue
        kind = str(row.get("patch_kind", "")).strip().lower()
        appended = _positive_int(row.get("patch_appended_count"))
        deleted = max(
            _positive_int(row.get("patch_deleted_count")),
            _positive_int(row.get("patch_removed_count")),
            _positive_int(row.get("support_patch_deleted_count")),
            _positive_int(row.get("support_patch_removed_count")),
        )
        if kind in {"append", "insert", "exchange", "swap"} or appended > 0:
            append_times.add(time_value)
        if kind in {"prune", "delete", "exchange", "swap"} or deleted > 0:
            prune_times.add(time_value)
    return tuple(sorted(append_times)), tuple(sorted(prune_times))


def _stabilization_spans(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[tuple[float, float], ...], int]:
    times = _numeric(rows, "time")
    active = np.asarray(
        [bool(row.get("solve_repair_applied")) for row in rows],
        dtype=bool,
    )
    indices = np.flatnonzero(active & np.isfinite(times))
    if indices.size == 0:
        return (), 0
    finite_times = times[np.isfinite(times)]
    dt = float(np.median(np.diff(finite_times))) if finite_times.size > 1 else 0.0
    groups: list[list[int]] = [[int(indices[0])]]
    for index in indices[1:]:
        if int(index) == groups[-1][-1] + 1:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    spans = tuple(
        (
            float(times[group[0]] - 0.5 * dt),
            float(times[group[-1]] + 0.5 * dt),
        )
        for group in groups
    )
    return spans, int(indices.size)


def _method_series(path: Path, *, label: str) -> MethodSeries:
    rows = _rows(_load_json(path), source_path=path)
    append_times, prune_times = _event_times(rows)
    stabilization_spans, stabilized_checkpoint_count = _stabilization_spans(rows)
    series = MethodSeries(
        label=label,
        source_path=Path(path),
        time=_numeric(rows, "time"),
        energy=_numeric(rows, "energy_expectation"),
        doublon=_numeric(rows, "doublon"),
        site_occupations=_site_occupations(rows, "site_occupations"),
        residual=_numeric(rows, "mclachlan_residual_ratio"),
        append_times=append_times,
        prune_times=prune_times,
        stabilization_spans=stabilization_spans,
        stabilized_checkpoint_count=stabilized_checkpoint_count,
    )
    for name, values in (
        ("time", series.time),
        ("energy", series.energy),
        ("doublon", series.doublon),
        ("site occupations", series.site_occupations),
        ("residual", series.residual),
    ):
        if not np.any(np.isfinite(values)):
            raise ValueError(f"{label} has no finite {name} values: {path}")
    return series


def _terminal_cost(path: Path | None, *, label: str) -> dict[str, Any]:
    if path is None:
        return {"status": "missing", "label": label}
    payload = _load_json(path)
    compile_result = dict(payload.get("compile_result", {}))
    rows = payload.get("rows", ())
    row = dict(rows[0]) if isinstance(rows, Sequence) and rows else {}
    values = {
        key: compile_result.get(key, row.get(key))
        for key in ("N2q", "D2q", "Dc")
    }
    if any(value is None for value in values.values()):
        raise ValueError(f"Terminal Qiskit cost is incomplete for {label}: {path}")
    return {
        "status": "ok",
        "label": label,
        "source_json": str(path),
        "source_sha256": _sha256(path),
        **{key: int(value) for key, value in values.items()},
    }


def load_comparison_case(case: ComparisonCase) -> LoadedComparison:
    ap_payload = _load_json(case.ap_json)
    ap_rows = _rows(ap_payload, source_path=case.ap_json)
    methods = (
        _method_series(case.ap_json, label="AP-McLachlan"),
        _method_series(case.avqds_json, label="AVQDS"),
        _method_series(case.avqds_t_json, label="AVQDS(T)"),
    )
    reference_time = methods[0].time
    for method in methods[1:]:
        if method.time.shape != reference_time.shape or not np.allclose(
            method.time,
            reference_time,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"Comparison time grids differ for {case.key}: "
                f"AP {reference_time.shape}, {method.label} {method.time.shape}."
            )
    return LoadedComparison(
        case=case,
        methods=methods,
        exact_energy=_numeric(ap_rows, "reference_energy"),
        seed_exact_energy=_numeric(ap_rows, "seed_reference_energy"),
        exact_doublon=_numeric(ap_rows, "doublon_exact"),
        seed_exact_doublon=_numeric(ap_rows, "seed_doublon_exact"),
        exact_site_occupations=_site_occupations(
            ap_rows, "site_occupations_exact"
        ),
        seed_exact_site_occupations=_site_occupations(
            ap_rows, "seed_site_occupations_exact"
        ),
        terminal_costs={
            "AP-McLachlan": _terminal_cost(
                case.ap_cost_json, label="AP-McLachlan"
            ),
            "AVQDS": _terminal_cost(case.avqds_cost_json, label="AVQDS"),
            "AVQDS(T)": _terminal_cost(
                case.avqds_t_cost_json, label="AVQDS(T)"
            ),
        },
    )


def _plot_finite(ax: Any, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.any(mask):
        ax.plot(x[mask], y[mask], **kwargs)


def _final_abs_error(values: np.ndarray, reference: np.ndarray) -> float | None:
    mask = np.isfinite(values) & np.isfinite(reference)
    if not np.any(mask):
        return None
    last = int(np.flatnonzero(mask)[-1])
    return float(abs(values[last] - reference[last]))


def _final_site_error(values: np.ndarray, reference: np.ndarray) -> float | None:
    if values.shape != reference.shape or values.ndim != 2:
        return None
    finite_rows = np.flatnonzero(
        np.any(np.isfinite(values) & np.isfinite(reference), axis=1)
    )
    if finite_rows.size == 0:
        return None
    last = int(finite_rows[-1])
    delta = np.abs(values[last] - reference[last])
    finite = delta[np.isfinite(delta)]
    return None if finite.size == 0 else float(np.max(finite))


def _final_value(values: np.ndarray) -> float | None:
    finite = np.flatnonzero(np.isfinite(values))
    return None if finite.size == 0 else float(values[int(finite[-1])])


def _nearest_finite_value(
    *,
    time: np.ndarray,
    values: np.ndarray,
    event_time: float,
) -> float | None:
    finite = np.flatnonzero(np.isfinite(time) & np.isfinite(values))
    if finite.size == 0:
        return None
    nearest = int(finite[np.argmin(np.abs(time[finite] - float(event_time)))])
    return float(values[nearest])


def _annotate_method_events(
    ax: Any,
    *,
    method: MethodSeries,
    values: np.ndarray,
) -> None:
    for start, stop in method.stabilization_spans:
        ax.axvspan(start, stop, color="#e1a62b", alpha=0.075, linewidth=0.0, zorder=0)
    for event_time in method.append_times:
        value = _nearest_finite_value(
            time=method.time,
            values=values,
            event_time=event_time,
        )
        if value is not None:
            ax.plot(
                [event_time],
                [value],
                marker="|",
                markersize=7.0,
                markeredgewidth=1.35,
                color="#1478d4",
                linestyle="none",
                zorder=5,
            )
    for event_time in method.prune_times:
        value = _nearest_finite_value(
            time=method.time,
            values=values,
            event_time=event_time,
        )
        if value is not None:
            ax.plot(
                [event_time],
                [value],
                marker="D",
                markersize=3.3,
                markeredgewidth=0.4,
                markeredgecolor="#9f1d20",
                markerfacecolor="#c62828",
                linestyle="none",
                zorder=6,
            )


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3g}"


def _format_count(value: Any) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def _case_manifest(case: LoadedComparison) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in case.methods:
        methods[method.label] = {
            "source_json": str(method.source_path),
            "source_sha256": _sha256(method.source_path),
            "point_count": int(method.time.size),
            "final_abs_energy_error_vs_ed": _final_abs_error(
                method.energy, case.exact_energy
            ),
            "final_abs_energy_error_vs_same_seed_exact": _final_abs_error(
                method.energy, case.seed_exact_energy
            ),
            "final_abs_doublon_error_vs_ed": _final_abs_error(
                method.doublon, case.exact_doublon
            ),
            "final_abs_doublon_error_vs_same_seed_exact": _final_abs_error(
                method.doublon, case.seed_exact_doublon
            ),
            "final_abs_site_occupations_error_max_vs_ed": _final_site_error(
                method.site_occupations, case.exact_site_occupations
            ),
            "final_abs_site_occupations_error_max_vs_same_seed_exact": (
                _final_site_error(
                    method.site_occupations,
                    case.seed_exact_site_occupations,
                )
            ),
            "final_native_residual_ratio": _final_value(method.residual),
            "append_event_count": int(len(method.append_times)),
            "prune_event_count": int(len(method.prune_times)),
            "stabilized_checkpoint_count": int(method.stabilized_checkpoint_count),
            "terminal_qiskit_cost": dict(case.terminal_costs[method.label]),
        }
    return {
        "key": case.case.key,
        "label": case.case.label,
        "time_start": float(case.methods[0].time[0]),
        "time_final": float(case.methods[0].time[-1]),
        "point_count": int(case.methods[0].time.size),
        "methods": methods,
    }


def build_comparison_page(
    *,
    cases: Sequence[ComparisonCase],
    output_pdf: Path,
    output_png: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("At least one comparison case is required.")
    loaded = tuple(load_comparison_case(case) for case in cases)
    figure = plt.figure(figsize=(11.0, 8.5))
    grid = figure.add_gridspec(
        2 * len(loaded),
        4,
        height_ratios=sum(([1.0, 0.23] for _ in loaded), []),
        hspace=0.48,
        wspace=0.22,
    )
    axes_by_column: list[Any] = [None, None, None, None]
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    for case_index, comparison in enumerate(loaded):
        plot_row = 2 * case_index
        axes: list[Any] = []
        for column in range(4):
            share_axis = axes_by_column[column]
            axis = figure.add_subplot(
                grid[plot_row, column],
                sharex=share_axis,
                sharey=share_axis,
            )
            if axes_by_column[column] is None:
                axes_by_column[column] = axis
            axes.append(axis)
        time = comparison.methods[0].time
        energy_axis, doublon_axis, site_axis, residual_axis = axes
        for method in comparison.methods:
            style = METHOD_STYLES[method.label]
            _plot_finite(
                energy_axis,
                method.time,
                method.energy,
                label=method.label,
                **style,
            )
            _plot_finite(
                doublon_axis,
                method.time,
                method.doublon,
                label=method.label,
                **style,
            )
            _plot_finite(
                site_axis,
                method.time,
                method.site_occupations[:, 0],
                **style,
            )
            site_one_style = dict(style)
            site_one_style["linestyle"] = "--"
            site_one_style["linewidth"] = 0.88 * float(style["linewidth"])
            _plot_finite(
                site_axis,
                method.time,
                method.site_occupations[:, 1],
                **site_one_style,
            )
            residual_mask = np.isfinite(method.time) & np.isfinite(method.residual)
            if np.any(residual_mask):
                residual_axis.semilogy(
                    method.time[residual_mask],
                    np.maximum(method.residual[residual_mask], 1.0e-16),
                    label=method.label,
                    **style,
                )
            _annotate_method_events(
                energy_axis,
                method=method,
                values=method.energy,
            )
            _annotate_method_events(
                doublon_axis,
                method=method,
                values=method.doublon,
            )
            _annotate_method_events(
                site_axis,
                method=method,
                values=method.site_occupations[:, 0],
            )
            _annotate_method_events(
                residual_axis,
                method=method,
                values=method.residual,
            )
        _plot_finite(
            energy_axis,
            time,
            comparison.exact_energy,
            label="ED exact",
            **EXACT_STYLE,
        )
        _plot_finite(
            energy_axis,
            time,
            comparison.seed_exact_energy,
            label="same-seed exact",
            **SEED_EXACT_STYLE,
        )
        _plot_finite(
            doublon_axis,
            time,
            comparison.exact_doublon,
            label="ED exact",
            **EXACT_STYLE,
        )
        _plot_finite(
            doublon_axis,
            time,
            comparison.seed_exact_doublon,
            label="same-seed exact",
            **SEED_EXACT_STYLE,
        )
        for site_index in range(2):
            exact_style = dict(EXACT_STYLE)
            seed_style = dict(SEED_EXACT_STYLE)
            if site_index == 1:
                exact_style["linestyle"] = "-."
                seed_style["linestyle"] = "--"
            _plot_finite(
                site_axis,
                time,
                comparison.exact_site_occupations[:, site_index],
                **exact_style,
            )
            _plot_finite(
                site_axis,
                time,
                comparison.seed_exact_site_occupations[:, site_index],
                **seed_style,
            )
        energy_axis.text(
            0.0,
            1.055,
            comparison.case.label,
            transform=energy_axis.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
        )
        for axis in axes:
            axis.grid(True, alpha=0.23, linewidth=0.6)
            axis.set_xlim(float(time[0]), float(time[-1]))
            axis.tick_params(labelsize=7)
            axis.set_xlabel("time", fontsize=8)
        energy_axis.set_ylabel("total energy", fontsize=8)
        doublon_axis.set_ylabel("doublon", fontsize=8)
        site_axis.set_ylabel("occupation", fontsize=8)
        residual_axis.set_ylabel("ratio", fontsize=8)
        if case_index == 0:
            energy_axis.set_title("Total energy", fontsize=10)
            doublon_axis.set_title("Doublon", fontsize=10)
            site_axis.set_title("Site occupations (solid: 0; dashed: 1)", fontsize=9)
            residual_axis.set_title("Method-native residual ratio", fontsize=10)
            legend_handles, legend_labels = energy_axis.get_legend_handles_labels()
        summary_axis = figure.add_subplot(grid[plot_row + 1, :])
        summary_axis.axis("off")
        parts = []
        for method in comparison.methods:
            parts.append(
                f"{method.label}: same-seed |dE|="
                f"{_format_metric(_final_abs_error(method.energy, comparison.seed_exact_energy))}, "
                f"|dD|={_format_metric(_final_abs_error(method.doublon, comparison.seed_exact_doublon))}, "
                f"|dn|max={_format_metric(_final_site_error(method.site_occupations, comparison.seed_exact_site_occupations))}; "
                f"ED |dE|={_format_metric(_final_abs_error(method.energy, comparison.exact_energy))}, "
                f"residual={_format_metric(_final_value(method.residual))}"
            )
        summary_axis.text(
            0.5,
            0.78,
            "\n".join(parts),
            ha="center",
            va="center",
            fontsize=6.2,
            linespacing=1.15,
        )
        event_parts = [
            f"{method.label}: append={len(method.append_times)}, "
            f"prune={len(method.prune_times)}, "
            f"stabilized checkpoints={method.stabilized_checkpoint_count}"
            for method in comparison.methods
        ]
        summary_axis.text(
            0.5,
            0.46,
            "   |   ".join(event_parts),
            ha="center",
            va="center",
            fontsize=6.5,
        )
        cost_parts = []
        for method in comparison.methods:
            cost = comparison.terminal_costs[method.label]
            cost_parts.append(
                f"{method.label}: N2q={_format_count(cost.get('N2q'))}, "
                f"D2q={_format_count(cost.get('D2q'))}, "
                f"Dc={_format_count(cost.get('Dc'))}"
            )
        summary_axis.text(
            0.5,
            0.14,
            "Terminal Qiskit cost: " + "   |   ".join(cost_parts),
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
        )
    figure.suptitle(
        "Same-seed comparison: APM, AVQDS Method 1, and AVQDS(T) Method 3",
        fontsize=13,
        y=0.975,
    )
    if legend_handles:
        figure.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.953),
            ncol=5,
            frameon=False,
            fontsize=8,
        )
    figure.text(
        0.5,
        0.012,
        "Residuals are method-native: APM realized McLachlan, AVQDS Method 1 "
        "continuous RHS, and AVQDS(T) Method 3 continuous-RHS TETRIS.\n"
        "Blue tick: append; red diamond: prune; amber band: stabilization. "
        "Exact curves are reporting-only.",
        ha="center",
        fontsize=6.7,
        linespacing=1.1,
        color="#444444",
    )
    figure.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.055)
    output_pdf = Path(output_pdf)
    output_png = Path(output_png)
    output_manifest = Path(output_manifest)
    for path in (output_pdf, output_png, output_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=190)
    figure.savefig(output_pdf)
    plt.close(figure)
    manifest = {
        "schema": "apm_avqds_method_comparison_page_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "exact_reference_scope": "post_run_reporting_overlay_only",
        "residual_comparison_scope": "method_native_not_common_projection",
        "output_pdf": str(output_pdf),
        "output_pdf_sha256": _sha256(output_pdf),
        "output_png": str(output_png),
        "output_png_sha256": _sha256(output_png),
        "comparison_cases": [_case_manifest(case) for case in loaded],
    }
    output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build same-seed AP-McLachlan/AVQDS/AVQDS(T) comparison plots."
    )
    parser.add_argument(
        "--case",
        nargs=5,
        action="append",
        metavar=("KEY", "LABEL", "AP_JSON", "AVQDS_JSON", "AVQDS_T_JSON"),
        required=True,
    )
    parser.add_argument(
        "--cost-case",
        nargs=4,
        action="append",
        metavar=("KEY", "AP_COST_JSON", "AVQDS_COST_JSON", "AVQDS_T_COST_JSON"),
        default=[],
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    costs_by_key = {
        str(raw[0]): (Path(raw[1]), Path(raw[2]), Path(raw[3]))
        for raw in args.cost_case
    }
    cases = tuple(
        ComparisonCase(
            key=str(raw[0]),
            label=str(raw[1]),
            ap_json=Path(raw[2]),
            avqds_json=Path(raw[3]),
            avqds_t_json=Path(raw[4]),
            ap_cost_json=(costs_by_key.get(str(raw[0])) or (None, None, None))[0],
            avqds_cost_json=(costs_by_key.get(str(raw[0])) or (None, None, None))[1],
            avqds_t_cost_json=(costs_by_key.get(str(raw[0])) or (None, None, None))[2],
        )
        for raw in args.case
    )
    manifest = build_comparison_page(
        cases=cases,
        output_pdf=args.output_pdf,
        output_png=args.output_png,
        output_manifest=args.output_manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

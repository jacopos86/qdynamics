"""Plot archive-style observables and block errors over the matched horizon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper5.stability.matrix_reference import closed_scalar_to_matrix_state


BLOCKS = ("rho", "B", "N", "A", "C")
BLOCK_LABELS = (r"$\rho$", r"$B$", r"$N$", r"$A$", r"$C$")
BLOCK_FIELDS = (
    "electron_density",
    "coherent_phonon",
    "phonon_density",
    "anomalous_phonon_density",
    "electron_phonon_correlation",
)
ENERGY_INDICES = {
    "electronic": 0,
    "phonon": 1,
    "electron_phonon": 2,
    "internal_total": 3,
}
COLORS = {
    "exact": "#171717",
    "raw": "#c4513c",
    "corrected": "#6f3c8f",
}
LINESTYLES = {"exact": "-", "raw": "--", "corrected": "-"}
LABELS = {
    "exact": "exact cutoff-16",
    "raw": "archive EOM",
    "corrected": "physicality-corrected",
}


def _load_lane(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        result = {key: payload[key].copy() for key in payload.files}
    result["metadata"] = json.loads(str(result.pop("metadata_json").item()))
    return result


def _interpolate_coordinates(
    target_times: np.ndarray,
    source_times: np.ndarray,
    source_coordinates: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(target_times, source_times, source_coordinates[:, index])
            for index in range(source_coordinates.shape[1])
        ]
    )


def _block_error_norms(
    approximate: np.ndarray,
    exact: np.ndarray,
) -> np.ndarray:
    errors = np.empty((approximate.shape[0], len(BLOCKS)), dtype=float)
    for row_index, (approximate_row, exact_row) in enumerate(
        zip(approximate, exact, strict=True)
    ):
        approximate_state = closed_scalar_to_matrix_state(approximate_row)
        exact_state = closed_scalar_to_matrix_state(exact_row)
        for block_index, field in enumerate(BLOCK_FIELDS):
            errors[row_index, block_index] = np.linalg.norm(
                getattr(approximate_state, field) - getattr(exact_state, field)
            )
    return errors


def _time_rms(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    duration = float(times[-1] - times[0])
    if duration <= 0.0:
        raise ValueError("trajectory duration must be positive")
    return np.sqrt(np.trapezoid(values**2, times, axis=0) / duration)


def _raw_termination_time(raw: dict[str, Any]) -> float:
    failure_time = raw["metadata"]["stats"]["failure_time"]
    if failure_time is None:
        return float(np.asarray(raw["times"], dtype=float)[-1])
    return float(failure_time)


def _set_padded_limits(axis: plt.Axes, values: list[np.ndarray]) -> None:
    finite = np.concatenate(
        [np.asarray(value, dtype=float)[np.isfinite(value)] for value in values]
    )
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = upper - lower
    padding = 0.08 * span if span > 0.0 else 0.08 * max(1.0, abs(lower))
    axis.set_ylim(lower - padding, upper + padding)


def _trajectory_metrics(
    exact: dict[str, Any],
    raw: dict[str, Any],
    corrected: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    exact_times = np.asarray(exact["times"], dtype=float)
    exact_coordinates = np.asarray(exact["coordinates"], dtype=float)
    dynamic = exact_coordinates - exact_coordinates[0]
    dynamic_norms = _block_error_norms(dynamic, np.zeros_like(dynamic))
    dynamic_scales = _time_rms(exact_times, dynamic_norms)
    if np.any(dynamic_scales <= 0.0):
        raise RuntimeError("an exact dynamic normalization scale is zero")

    corrected_times = np.asarray(corrected["times"], dtype=float)
    corrected_exact = _interpolate_coordinates(
        corrected_times,
        exact_times,
        exact_coordinates,
    )
    corrected_errors = _block_error_norms(
        np.asarray(corrected["coordinates"], dtype=float),
        corrected_exact,
    ) / dynamic_scales

    raw_times = np.asarray(raw["times"], dtype=float)
    raw_exact = _interpolate_coordinates(raw_times, exact_times, exact_coordinates)
    raw_errors = _block_error_norms(
        np.asarray(raw["coordinates"], dtype=float),
        raw_exact,
    ) / dynamic_scales

    overlap_end = float(raw_times[-1])
    corrected_overlap_mask = corrected_times <= overlap_end + 1e-12
    corrected_overlap_times = corrected_times[corrected_overlap_mask]
    corrected_overlap_errors = corrected_errors[corrected_overlap_mask]
    raw_metadata = raw["metadata"]
    metrics = {
        "normalization_horizon": float(exact_times[-1]),
        "dynamic_scales": {
            block: float(dynamic_scales[index])
            for index, block in enumerate(BLOCKS)
        },
        "raw_failure_time": raw_metadata["stats"]["failure_time"],
        "raw_recorded_end": overlap_end,
        "raw_overlap_time_rms": {
            block: float(value)
            for block, value in zip(
                BLOCKS,
                _time_rms(raw_times, raw_errors),
                strict=True,
            )
        },
        "corrected_overlap_time_rms": {
            block: float(value)
            for block, value in zip(
                BLOCKS,
                _time_rms(corrected_overlap_times, corrected_overlap_errors),
                strict=True,
            )
        },
        "corrected_full_time_rms": {
            block: float(value)
            for block, value in zip(
                BLOCKS,
                _time_rms(corrected_times, corrected_errors),
                strict=True,
            )
        },
    }
    return metrics, raw_errors, corrected_errors


def _style_axis(
    axis: plt.Axes,
    failure_time: float,
    final_time: float,
    *,
    expand_early_time: bool = False,
) -> None:
    axis.axvline(
        failure_time,
        color=COLORS["raw"],
        linewidth=0.7,
        alpha=0.45,
        zorder=0,
    )
    axis.grid(alpha=0.17, linewidth=0.5)
    axis.set_xlim(0.0, final_time)
    if expand_early_time:
        axis.set_xscale("symlog", linthresh=4.0, linscale=1.0)
        ticks = [value for value in (0.0, 4.0, 10.0, 100.0, 1000.0) if value <= final_time]
        axis.set_xticks(ticks)
        axis.set_xticklabels([f"{value:g}" for value in ticks])


def _plot_archive_observables(
    exact: dict[str, Any],
    raw: dict[str, Any],
    corrected: dict[str, Any],
    output_stem: Path,
) -> None:
    failure_time = _raw_termination_time(raw)
    final_time = float(np.asarray(exact["times"], dtype=float)[-1])
    figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.55), sharex=False)
    panels = (
        ("occupation", None, r"site occupation"),
        ("occupation_early", None, r"early response"),
        ("energy", "electronic", r"electronic energy"),
        ("energy", "phonon", r"phonon energy"),
        ("energy", "electron_phonon", "electron–phonon energy"),
        ("energy", "internal_total", r"total internal energy"),
    )
    lanes = {"exact": exact, "raw": raw, "corrected": corrected}
    for panel_index, (kind, component, title) in enumerate(panels):
        axis = axes.flat[panel_index]
        plotted_values: dict[str, np.ndarray] = {}
        for lane, payload in lanes.items():
            times = np.asarray(payload["times"], dtype=float)
            if kind.startswith("occupation"):
                values = np.asarray(payload["site_occupation"], dtype=float)
            else:
                values = np.asarray(payload["energy_components"], dtype=float)[
                    :, ENERGY_INDICES[str(component)]
                ]
            plotted_values[lane] = values
            axis.plot(
                times,
                values,
                color=COLORS[lane],
                linestyle=LINESTYLES[lane],
                linewidth={"exact": 1.0, "raw": 0.65, "corrected": 0.85}[lane],
                alpha=0.62 if lane == "raw" else 0.92,
                label=LABELS[lane],
            )
        axis.set_title(f"({chr(97 + panel_index)}) {title}", loc="left")
        axis.grid(alpha=0.17, linewidth=0.5)
        if kind == "occupation_early":
            axis.set_xlim(0.0, 20.0)
            axis.axvspan(0.0, 4.0, color="#b9d7e8", alpha=0.22)
            axis.set_xlabel(r"$t\,t_{\rm hop}$")
            early_values = []
            for lane, payload in lanes.items():
                times = np.asarray(payload["times"], dtype=float)
                early_values.append(plotted_values[lane][times <= 20.0])
            _set_padded_limits(axis, early_values)
        else:
            _style_axis(axis, failure_time, final_time)
            if panel_index >= 3:
                axis.set_xlabel(r"$t\,t_{\rm hop}$")
            if kind == "occupation":
                axis.set_ylim(-0.02, 1.02)
            elif component == "internal_total":
                _set_padded_limits(axis, list(plotted_values.values()))
            else:
                _set_padded_limits(
                    axis,
                    [plotted_values["exact"], plotted_values["corrected"]],
                )
        if kind == "occupation":
            axis.set_ylabel(r"$\rho_{11}$")
        elif panel_index in (2, 3):
            axis.set_ylabel(r"energy / $t_{\rm hop}$")
    axes.flat[0].text(
        failure_time + 0.025 * final_time,
        0.95,
        "raw curve leaves\nphysical scale",
        color=COLORS["raw"],
        va="top",
        fontsize=6.2,
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.15,
        handlelength=2.2,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.105,
        top=0.885,
        hspace=0.42,
        wspace=0.32,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_archive_observables_short(
    exact: dict[str, Any],
    raw: dict[str, Any],
    corrected: dict[str, Any],
    output_stem: Path,
    *,
    horizon: float = 20.0,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.55), sharex=True)
    panels = (
        ("occupation", None, "site occupation"),
        ("energy", "electronic", "electronic energy"),
        ("energy", "phonon", "phonon energy"),
        ("energy", "electron_phonon", "electron–phonon energy"),
        ("energy", "internal_total", "total internal energy"),
    )
    lanes = {"exact": exact, "raw": raw, "corrected": corrected}
    for panel_index, (kind, component, title) in enumerate(panels):
        axis = axes.flat[panel_index]
        short_values: list[np.ndarray] = []
        for lane, payload in lanes.items():
            times = np.asarray(payload["times"], dtype=float)
            mask = times <= horizon + 1e-12
            if kind == "occupation":
                values = np.asarray(payload["site_occupation"], dtype=float)
            else:
                values = np.asarray(payload["energy_components"], dtype=float)[
                    :, ENERGY_INDICES[str(component)]
                ]
            short_values.append(values[mask])
            axis.plot(
                times[mask],
                values[mask],
                color=COLORS[lane],
                linestyle=LINESTYLES[lane],
                linewidth={"exact": 1.0, "raw": 0.75, "corrected": 0.9}[lane],
                alpha=0.68 if lane == "raw" else 0.94,
                label=LABELS[lane],
            )
        axis.set_title(f"({chr(97 + panel_index)}) {title}", loc="left")
        axis.set_xlim(0.0, horizon)
        axis.axvspan(0.0, 4.0, color="#b9d7e8", alpha=0.20)
        axis.grid(alpha=0.17, linewidth=0.5)
        if panel_index >= 3:
            axis.set_xlabel(r"$t\,t_{\rm hop}$")
        if kind == "occupation":
            axis.set_ylabel(r"$\rho_{11}$")
            axis.set_ylim(-0.02, 1.02)
        else:
            axis.set_ylabel(r"energy / $t_{\rm hop}$")
            _set_padded_limits(axis, short_values)

    figure.delaxes(axes.flat[5])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.15,
        handlelength=2.2,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.105,
        top=0.885,
        hspace=0.42,
        wspace=0.32,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_block_errors(
    exact: dict[str, Any],
    raw: dict[str, Any],
    corrected: dict[str, Any],
    raw_errors: np.ndarray,
    corrected_errors: np.ndarray,
    output_stem: Path,
) -> None:
    raw_times = np.asarray(raw["times"], dtype=float)
    corrected_times = np.asarray(corrected["times"], dtype=float)
    failure_time = _raw_termination_time(raw)
    final_time = float(np.asarray(exact["times"], dtype=float)[-1])
    figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.55))
    for block_index, block_label in enumerate(BLOCK_LABELS):
        axis = axes.flat[block_index]
        raw_values = np.where(
            raw_errors[:, block_index] > 1e-8,
            raw_errors[:, block_index],
            np.nan,
        )
        corrected_values = np.where(
            corrected_errors[:, block_index] > 1e-8,
            corrected_errors[:, block_index],
            np.nan,
        )
        axis.plot(
            raw_times,
            raw_values,
            color=COLORS["raw"],
            linestyle=LINESTYLES["raw"],
            linewidth=0.85,
            label=LABELS["raw"],
        )
        axis.plot(
            corrected_times,
            corrected_values,
            color=COLORS["corrected"],
            linewidth=0.85,
            label=LABELS["corrected"],
        )
        axis.set_yscale("log")
        axis.set_title(
            f"({chr(97 + block_index)}) {block_label} block",
            loc="left",
        )
        if block_index >= 3:
            axis.set_xlabel(r"$t\,t_{\rm hop}$")
        _style_axis(
            axis,
            failure_time,
            final_time,
            expand_early_time=True,
        )

    axis = axes.flat[5]
    for lane, payload in (("exact", exact), ("raw", raw), ("corrected", corrected)):
        times = np.asarray(payload["times"], dtype=float)
        maximum = np.max(np.abs(np.asarray(payload["coordinates"], dtype=float)), axis=1)
        axis.plot(
            times,
            maximum,
            color=COLORS[lane],
            linestyle=LINESTYLES[lane],
            linewidth=0.9,
            label=LABELS[lane],
        )
    axis.axhline(1e4, color=COLORS["raw"], linewidth=0.7, linestyle=":")
    axis.set_yscale("log")
    axis.set_title(r"(f) amplitude and termination", loc="left")
    axis.set_ylabel(r"$\max_j|x_j(t)|$")
    axis.set_xlabel(r"$t\,t_{\rm hop}$")
    _style_axis(
        axis,
        failure_time,
        final_time,
        expand_early_time=True,
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.3,
        handlelength=2.2,
    )
    figure.text(
        0.012,
        0.52,
        r"instantaneous normalized block error $e_Y(t)$",
        rotation=90,
        va="center",
        ha="left",
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.105,
        top=0.885,
        hspace=0.42,
        wspace=0.30,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_block_errors_short(
    exact: dict[str, Any],
    raw: dict[str, Any],
    corrected: dict[str, Any],
    raw_errors: np.ndarray,
    corrected_errors: np.ndarray,
    output_stem: Path,
    *,
    horizon: float = 20.0,
) -> None:
    raw_times = np.asarray(raw["times"], dtype=float)
    corrected_times = np.asarray(corrected["times"], dtype=float)
    raw_mask = raw_times <= horizon + 1e-12
    corrected_mask = corrected_times <= horizon + 1e-12

    figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.55), sharex=True)
    for block_index, block_label in enumerate(BLOCK_LABELS):
        axis = axes.flat[block_index]
        raw_values = np.where(
            raw_errors[raw_mask, block_index] > 1e-8,
            raw_errors[raw_mask, block_index],
            np.nan,
        )
        corrected_values = np.where(
            corrected_errors[corrected_mask, block_index] > 1e-8,
            corrected_errors[corrected_mask, block_index],
            np.nan,
        )
        axis.plot(
            raw_times[raw_mask],
            raw_values,
            color=COLORS["raw"],
            linestyle=LINESTYLES["raw"],
            linewidth=0.9,
            label=LABELS["raw"],
        )
        axis.plot(
            corrected_times[corrected_mask],
            corrected_values,
            color=COLORS["corrected"],
            linewidth=0.9,
            label=LABELS["corrected"],
        )
        axis.set_yscale("log")
        axis.set_title(
            f"({chr(97 + block_index)}) {block_label} block",
            loc="left",
        )
        axis.set_xlim(0.0, horizon)
        axis.axvspan(0.0, 4.0, color="#b9d7e8", alpha=0.20)
        axis.grid(alpha=0.17, linewidth=0.5)
        if block_index >= 3:
            axis.set_xlabel(r"$t\,t_{\rm hop}$")

    axis = axes.flat[5]
    amplitude_values: list[np.ndarray] = []
    for lane, payload in (("exact", exact), ("raw", raw), ("corrected", corrected)):
        times = np.asarray(payload["times"], dtype=float)
        mask = times <= horizon + 1e-12
        maximum = np.max(
            np.abs(np.asarray(payload["coordinates"], dtype=float)),
            axis=1,
        )
        amplitude_values.append(maximum[mask])
        axis.plot(
            times[mask],
            maximum[mask],
            color=COLORS[lane],
            linestyle=LINESTYLES[lane],
            linewidth=0.9,
            label=LABELS[lane],
        )
    axis.set_title(r"(f) coordinate amplitude", loc="left")
    axis.set_ylabel(r"$\max_j|x_j(t)|$")
    axis.set_xlabel(r"$t\,t_{\rm hop}$")
    axis.set_xlim(0.0, horizon)
    axis.axvspan(0.0, 4.0, color="#b9d7e8", alpha=0.20)
    axis.grid(alpha=0.17, linewidth=0.5)
    _set_padded_limits(axis, amplitude_values)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.3,
        handlelength=2.2,
    )
    figure.text(
        0.012,
        0.52,
        r"instantaneous normalized block error $e_Y(t)$",
        rotation=90,
        va="center",
        ha="left",
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.105,
        top=0.885,
        hspace=0.42,
        wspace=0.30,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_figures(
    trajectory_dir: Path,
    output_dir: Path,
    *,
    raw_filename: str = "raw_trajectory.npz",
) -> None:
    plt.rcParams.update(
        {
            "font.size": 6.8,
            "axes.titlesize": 7.4,
            "axes.labelsize": 6.9,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "lines.solid_capstyle": "round",
        }
    )
    exact = _load_lane(trajectory_dir / "exact_trajectory.npz")
    raw = _load_lane(trajectory_dir / raw_filename)
    corrected = _load_lane(trajectory_dir / "corrected_trajectory.npz")
    metrics, raw_errors, corrected_errors = _trajectory_metrics(
        exact,
        raw,
        corrected,
    )
    _plot_archive_observables(
        exact,
        raw,
        corrected,
        output_dir / "paper_v_archive_observable_trajectories",
    )
    _plot_archive_observables_short(
        exact,
        raw,
        corrected,
        output_dir / "paper_v_archive_observable_trajectories_short",
    )
    _plot_block_errors(
        exact,
        raw,
        corrected,
        raw_errors,
        corrected_errors,
        output_dir / "paper_v_block_error_trajectories",
    )
    _plot_block_errors_short(
        exact,
        raw,
        corrected,
        raw_errors,
        corrected_errors,
        output_dir / "paper_v_block_error_trajectories_short",
    )
    (output_dir / "paper_v_long_horizon_trajectory_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--raw-filename", default="raw_trajectory.npz")
    args = parser.parse_args()
    build_figures(
        args.trajectory_dir,
        args.output_dir,
        raw_filename=args.raw_filename,
    )


if __name__ == "__main__":
    main()

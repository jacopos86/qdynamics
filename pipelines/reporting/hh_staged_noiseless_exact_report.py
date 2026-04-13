#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from docs.reports.pdf_utils import (
    current_command_string,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_compact_table,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)
from src.quantum.drives_time_potential import default_spatial_weights, evaluate_drive_waveform


"series = {t_i, F_i, E_exact_i, E_method_i, O_i}_{i=1..N}"
@dataclass(frozen=True)
class NoiselessSeries:
    times: np.ndarray
    fidelity: np.ndarray
    energy_total_exact: np.ndarray
    energy_total_method: np.ndarray
    abs_energy_total_error: np.ndarray
    staggered_exact: np.ndarray
    staggered_method: np.ndarray
    doublon_exact: np.ndarray
    doublon_method: np.ndarray


"payload = json(path)"
def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}, got {type(payload).__name__}.")
    return payload


"x = float(value) if finite else nan"
def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


"method = first staged dynamics method"
def _default_method_name(payload: Mapping[str, Any]) -> str:
    settings = payload.get("settings", {})
    if not isinstance(settings, Mapping):
        raise KeyError("Missing settings block in workflow payload.")
    dynamics = settings.get("dynamics", {})
    if not isinstance(dynamics, Mapping):
        raise KeyError("Missing settings.dynamics block in workflow payload.")
    methods = dynamics.get("methods", [])
    if not isinstance(methods, Sequence) or not methods:
        raise KeyError("Missing settings.dynamics.methods in workflow payload.")
    return str(methods[0])


"rows = payload.dynamics_noiseless.profiles[profile].methods[method].trajectory"
def _trajectory_rows(payload: Mapping[str, Any], *, profile: str, method: str) -> list[Mapping[str, Any]]:
    dynamics = payload.get("dynamics_noiseless", {})
    if not isinstance(dynamics, Mapping):
        raise KeyError("Missing dynamics_noiseless in workflow payload.")
    profiles = dynamics.get("profiles", {})
    if not isinstance(profiles, Mapping):
        raise KeyError("Missing dynamics_noiseless.profiles in workflow payload.")
    profile_payload = profiles.get(profile, {})
    if not isinstance(profile_payload, Mapping):
        raise KeyError(f"Missing profile {profile!r} in dynamics_noiseless.profiles.")
    methods = profile_payload.get("methods", {})
    if not isinstance(methods, Mapping):
        raise KeyError(f"Missing methods block for profile {profile!r}.")
    method_payload = methods.get(method, {})
    if not isinstance(method_payload, Mapping):
        raise KeyError(f"Missing method {method!r} under profile {profile!r}.")
    rows = method_payload.get("trajectory", [])
    if not isinstance(rows, list) or not rows:
        raise KeyError(f"Missing non-empty trajectory for {profile}/{method}.")
    clean_rows = [row for row in rows if isinstance(row, Mapping)]
    if not clean_rows:
        raise TypeError(f"Trajectory rows for {profile}/{method} must be mappings.")
    return clean_rows


"arr_i = row_i[key]"
def _array_from_rows(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_float_or_nan(row.get(key)) for row in rows], dtype=float)


"series = parse(rows)"
def load_noiseless_series(payload: Mapping[str, Any], *, profile: str = "drive", method: str | None = None) -> NoiselessSeries:
    chosen_method = _default_method_name(payload) if method is None else str(method)
    rows = _trajectory_rows(payload, profile=str(profile), method=chosen_method)
    times = _array_from_rows(rows, "time")
    energy_total_exact = _array_from_rows(rows, "energy_total_exact")
    energy_total_method = _array_from_rows(rows, "energy_total_trotter")
    staggered_exact = _array_from_rows(rows, "staggered_exact")
    staggered_method = _array_from_rows(rows, "staggered_trotter")
    doublon_exact = _array_from_rows(rows, "doublon_exact")
    doublon_method = _array_from_rows(rows, "doublon_trotter")
    fidelity = _array_from_rows(rows, "fidelity")
    return NoiselessSeries(
        times=times,
        fidelity=fidelity,
        energy_total_exact=energy_total_exact,
        energy_total_method=energy_total_method,
        abs_energy_total_error=np.abs(energy_total_method - energy_total_exact),
        staggered_exact=staggered_exact,
        staggered_method=staggered_method,
        doublon_exact=doublon_exact,
        doublon_method=doublon_method,
    )


"ylim_fidelity = [max(0, min(F)-pad), 1]"
def _set_fidelity_ylim(ax: Any, fidelity: np.ndarray) -> None:
    finite = fidelity[np.isfinite(fidelity)]
    if finite.size == 0:
        ax.set_ylim(0.0, 1.0)
        return
    lower = float(np.min(finite))
    span = max(1.0 - lower, 0.05)
    ax.set_ylim(max(0.0, lower - 0.08 * span), 1.0)


"waveform_page = v(t), s_j v(t)"
def _render_waveform_page(pdf: Any, *, physics: Mapping[str, Any], dynamics: Mapping[str, Any], times: np.ndarray) -> None:
    require_matplotlib()
    plt = get_plt()
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), sharex=True)
    drive_cfg = {
        "drive_omega": float(dynamics.get("drive_omega", 1.0)),
        "drive_tbar": float(dynamics.get("drive_tbar", 1.0)),
        "drive_phi": float(dynamics.get("drive_phi", 0.0)),
        "drive_t0": float(dynamics.get("drive_t0", 0.0)),
    }
    amplitude = float(dynamics.get("drive_A", 0.0))
    waveform = evaluate_drive_waveform(times, drive_cfg, amplitude=amplitude)
    n_sites = int(physics.get("L", 2))
    weights = default_spatial_weights(
        n_sites,
        mode=str(dynamics.get("drive_pattern", "staggered")),
        custom=dynamics.get("drive_custom_s"),
    )
    axes[0].plot(times, waveform, color="tab:blue", linewidth=2.0)
    axes[0].set_title("Driven scalar waveform v(t)")
    axes[0].set_ylabel("v(t)")
    axes[0].grid(True, alpha=0.3)
    for site, weight in enumerate(weights):
        axes[1].plot(times, float(weight) * waveform, linewidth=2.0, label=f"site {site}")
    axes[1].set_title("Site-resolved onsite-density drive s_j v(t)")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("potential")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


"trajectory_page = exact(t) vs method(t), fidelity(method, exact)"
def _render_primary_page(pdf: Any, *, series: NoiselessSeries, method: str) -> None:
    require_matplotlib()
    plt = get_plt()
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), sharex=True)
    axes[0].plot(series.times, series.energy_total_exact, linewidth=2.4, color="black", label="exact")
    axes[0].plot(series.times, series.energy_total_method, linewidth=2.0, linestyle="--", color="tab:blue", label=method)
    axes[0].set_title("Driven total energy vs exact reference")
    axes[0].set_ylabel("E_total")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(series.times, series.fidelity, linewidth=2.0, color="tab:green", label=f"{method} fidelity to exact")
    axes[1].set_title("Driven fidelity to exact state")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("fidelity")
    _set_fidelity_ylim(axes[1], series.fidelity)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


"observables_page = {O_exact(t), O_method(t), |ΔE|(t)}"
def _render_observables_page(pdf: Any, *, series: NoiselessSeries, method: str) -> None:
    require_matplotlib()
    plt = get_plt()
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.5), sharex=True)
    axes[0].plot(series.times, series.staggered_exact, linewidth=2.4, color="black", label="exact")
    axes[0].plot(series.times, series.staggered_method, linewidth=2.0, linestyle="--", color="tab:purple", label=method)
    axes[0].set_title("Driven staggered density vs exact reference")
    axes[0].set_ylabel("staggered")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(series.times, series.doublon_exact, linewidth=2.4, color="black", label="exact")
    axes[1].plot(series.times, series.doublon_method, linewidth=2.0, linestyle="--", color="tab:orange", label=method)
    axes[1].set_title("Driven doublon vs exact reference")
    axes[1].set_ylabel("doublon")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    axes[2].plot(series.times, series.abs_energy_total_error, linewidth=2.0, color="tab:red")
    axes[2].set_title("|E_total(method) - E_total(exact)|")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("abs error")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


"pdf = render(payload, profile, method)"
def write_noiseless_exact_report(
    *,
    input_json: str | Path,
    output_pdf: str | Path,
    profile: str = "drive",
    method: str | None = None,
    run_command: str | None = None,
) -> Path:
    payload = _read_json(Path(input_json))
    if str(payload.get("pipeline")) != "hh_staged_noiseless":
        raise ValueError("This report expects an hh_staged_noiseless workflow JSON.")
    settings = payload.get("settings", {})
    if not isinstance(settings, Mapping):
        raise KeyError("Missing settings block in workflow payload.")
    physics = settings.get("physics", {})
    dynamics = settings.get("dynamics", {})
    if not isinstance(physics, Mapping) or not isinstance(dynamics, Mapping):
        raise KeyError("Missing settings.physics or settings.dynamics in workflow payload.")
    chosen_method = _default_method_name(payload) if method is None else str(method)
    series = load_noiseless_series(payload, profile=str(profile), method=chosen_method)

    require_matplotlib()
    output_path = Path(output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PdfPages = get_PdfPages()
    command_str = current_command_string() if run_command is None else str(run_command)

    with PdfPages(str(output_path)) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein",
            ansatz="staged HH replay baseline compared to exact driven reference",
            drive_enabled=bool(dynamics.get("enable_drive", False)),
            t=float(physics.get("t", float("nan"))),
            U=float(physics.get("u", float("nan"))),
            dv=float(physics.get("dv", float("nan"))),
            extra={
                "L": int(physics.get("L", 0)),
                "omega0": float(physics.get("omega0", float("nan"))),
                "g_ep": float(physics.get("g_ep", float("nan"))),
                "n_ph_max": int(physics.get("n_ph_max", 0)),
                "boundary": str(physics.get("boundary", "")),
                "ordering": str(physics.get("ordering", "")),
                "profile": str(profile),
                "method": str(chosen_method),
                "t_final": float(dynamics.get("t_final", float("nan"))),
                "num_times": int(len(series.times)),
                "trotter_steps": int(dynamics.get("trotter_steps", 0)),
                "exact_steps_multiplier": int(dynamics.get("exact_steps_multiplier", 0)),
                "drive_pattern": str(dynamics.get("drive_pattern", "")),
                "drive_A": float(dynamics.get("drive_A", float("nan"))),
                "drive_omega": float(dynamics.get("drive_omega", float("nan"))),
                "drive_tbar": float(dynamics.get("drive_tbar", float("nan"))),
                "drive_phi": float(dynamics.get("drive_phi", float("nan"))),
                "drive_t0": float(dynamics.get("drive_t0", float("nan"))),
            },
            command=command_str,
        )
        summary_lines = [
            "Driven HH noiseless exact-comparison report",
            "",
            f"Input JSON: {Path(input_json).resolve()}",
            f"Profile/method: {profile}/{chosen_method}",
            "The energy and observable panels compare the propagated method directly against the driven exact reference.",
            "The fidelity panel is the propagated-state fidelity to the exact driven state; there is no separate oscillatory exact-fidelity line because exact-vs-exact would be identically 1.",
        ]
        render_text_page(pdf, summary_lines, fontsize=10, line_spacing=0.03)

        fig, ax = get_plt().subplots(figsize=(11.0, 4.8))
        render_compact_table(
            ax,
            title="Driven noiseless summary",
            col_labels=[
                "Final fidelity",
                "Min fidelity",
                "Max |ΔE_total|",
                "Final |ΔE_total|",
                "Max |Δ staggered|",
                "Max |Δ doublon|",
            ],
            rows=[[
                f"{float(series.fidelity[-1]):.9f}",
                f"{float(np.nanmin(series.fidelity)):.9f}",
                f"{float(np.nanmax(series.abs_energy_total_error)):.3e}",
                f"{float(series.abs_energy_total_error[-1]):.3e}",
                f"{float(np.nanmax(np.abs(series.staggered_method - series.staggered_exact))):.3e}",
                f"{float(np.nanmax(np.abs(series.doublon_method - series.doublon_exact))):.3e}",
            ]],
            fontsize=9,
        )
        pdf.savefig(fig)
        get_plt().close(fig)

        if bool(dynamics.get("enable_drive", False)):
            _render_waveform_page(pdf, physics=physics, dynamics=dynamics, times=series.times)
        _render_primary_page(pdf, series=series, method=chosen_method)
        _render_observables_page(pdf, series=series, method=chosen_method)
        render_command_page(
            pdf,
            command_str,
            script_name="pipelines/reporting/hh_staged_noiseless_exact_report.py",
            extra_header_lines=[
                f"input_json: {Path(input_json).resolve()}",
                f"output_pdf: {output_path.resolve()}",
            ],
        )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a focused PDF comparing driven staged-noiseless dynamics against the exact reference."
    )
    parser.add_argument("--input-json", required=True, help="Path to an hh_staged_noiseless workflow JSON.")
    parser.add_argument("--output-pdf", required=True, help="Output PDF path.")
    parser.add_argument("--profile", default="drive", help="Noiseless profile name. Default: drive")
    parser.add_argument("--method", default=None, help="Method name. Default: first configured method")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_path = write_noiseless_exact_report(
        input_json=args.input_json,
        output_pdf=args.output_pdf,
        profile=str(args.profile),
        method=args.method,
    )
    print(f"workflow_pdf={output_path}")


if __name__ == "__main__":
    main()

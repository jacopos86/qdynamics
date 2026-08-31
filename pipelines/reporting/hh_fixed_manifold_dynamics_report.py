#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

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


_REPO_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_TRAJECTORY_KEYS: tuple[tuple[str | None, str], ...] = (
    (None, "time"),
    ("audit", "fidelity_exact_audit"),
    ("audit", "energy_ansatz_exact_audit"),
    ("audit", "energy_reference_exact_audit"),
    ("audit", "abs_energy_total_error_exact_audit"),
    ("geometry", "rho_miss"),
    ("geometry", "theta_dot_l2"),
    ("geometry", "condition_number"),
)
_REQUIRED_SOURCE_SETTING_KEYS: tuple[str, ...] = (
    "L",
    "t",
    "u",
    "dv",
    "omega0",
    "g_ep",
    "n_ph_max",
    "boundary",
    "ordering",
)


"entries = [load_run_entry(path_j)]_{j=1..m}"
@dataclass(frozen=True)
class ReportEntry:
    json_path: Path
    payload: dict[str, Any]
    source_artifact_path: Path
    source_settings: dict[str, Any]
    label: str
    times: np.ndarray
    fidelity: np.ndarray
    energy_ansatz: np.ndarray
    energy_reference: np.ndarray
    abs_energy_error: np.ndarray
    rho_miss: np.ndarray
    theta_dot_l2: np.ndarray
    condition_number: np.ndarray


"payload = json(path)"
def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


"resolved_path = raw if absolute else unique_existing(repo_root/raw, base_parent/raw)"
def _resolve_path(raw: str | Path, *, base_parent: Path) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {candidate}")
        return candidate
    probes = [
        (_REPO_ROOT / candidate).resolve(),
        (base_parent / candidate).resolve(),
    ]
    existing: list[Path] = []
    for probe in probes:
        if probe.exists() and probe not in existing:
            existing.append(probe)
    if len(existing) > 1:
        raise ValueError(
            f"Ambiguous relative artifact path {raw!r}; matches: {', '.join(str(p) for p in existing)}"
        )
    if not existing:
        raise FileNotFoundError(
            f"Could not resolve relative artifact path {raw!r} against repo root {_REPO_ROOT} or {base_parent}."
        )
    return existing[0]


"series_i = metric(trajectory_i)"
def _trajectory_array(
    payload: Mapping[str, Any],
    section: str | None,
    key: str,
    *,
    fallback: float = float("nan"),
    required: bool = False,
) -> np.ndarray:
    trajectory = payload.get("trajectory", [])
    if not isinstance(trajectory, list):
        raise TypeError("trajectory must be a list.")
    values: list[float] = []
    for idx, row in enumerate(trajectory):
        if not isinstance(row, Mapping):
            raise TypeError("trajectory rows must be mappings.")
        if section is None:
            if required and key not in row:
                raise KeyError(f"Missing trajectory key {key!r} at row {idx}.")
            values.append(float(row.get(key, fallback)))
            continue
        source: Any = row.get(section)
        if not isinstance(source, Mapping):
            if required:
                raise KeyError(f"Missing trajectory section {section!r} at row {idx}.")
            values.append(float(fallback))
            continue
        if required and key not in source:
            raise KeyError(f"Missing trajectory key {section}.{key!r} at row {idx}.")
        values.append(float(source.get(key, fallback)))
    return np.asarray(values, dtype=float)


"entry = measured_payload ⊕ source_settings ⊕ plotted_series"
def load_report_entry(json_path: str | Path) -> ReportEntry:
    path = Path(json_path).resolve()
    payload = _read_json(path)
    pipeline_name = str(payload.get("pipeline", ""))
    if pipeline_name != "hh_fixed_manifold_measured_mclachlan_v1":
        raise ValueError(
            f"Expected fixed-manifold measured payload, got pipeline={pipeline_name!r} for {path}."
        )
    source_raw = payload.get("input_artifact_json")
    if not source_raw:
        raise KeyError(f"Missing input_artifact_json in {path}.")
    source_artifact_path = _resolve_path(str(source_raw), base_parent=path.parent)
    source_payload = _read_json(source_artifact_path)
    source_settings = dict(source_payload.get("settings", {}))
    if not source_settings:
        raise KeyError(f"Missing source settings in {source_artifact_path}.")
    missing_source_keys = [key for key in _REQUIRED_SOURCE_SETTING_KEYS if key not in source_settings]
    if missing_source_keys:
        raise KeyError(
            f"Missing source setting keys in {source_artifact_path}: {', '.join(missing_source_keys)}"
        )

    loader = payload.get("loader", {}) if isinstance(payload.get("loader", {}), Mapping) else {}
    summary = payload.get("summary", {}) if isinstance(payload.get("summary", {}), Mapping) else {}
    label = str(
        payload.get("run_name")
        or loader.get("resolved_family")
        or loader.get("fixed_scaffold_kind")
        or path.stem
    )
    for section, key in _REQUIRED_TRAJECTORY_KEYS:
        _trajectory_array(payload, section, key, required=True)
    times = _trajectory_array(payload, None, "time", required=True)
    if times.size == 0:
        raise ValueError(f"Cannot render report for {path}: trajectory is empty.")
    return ReportEntry(
        json_path=path,
        payload=payload,
        source_artifact_path=source_artifact_path,
        source_settings=source_settings,
        label=label,
        times=times,
        fidelity=_trajectory_array(payload, "audit", "fidelity_exact_audit", required=True),
        energy_ansatz=_trajectory_array(payload, "audit", "energy_ansatz_exact_audit", required=True),
        energy_reference=_trajectory_array(payload, "audit", "energy_reference_exact_audit", required=True),
        abs_energy_error=_trajectory_array(payload, "audit", "abs_energy_total_error_exact_audit", required=True),
        rho_miss=_trajectory_array(payload, "geometry", "rho_miss", required=True),
        theta_dot_l2=_trajectory_array(payload, "geometry", "theta_dot_l2", required=True),
        condition_number=_trajectory_array(payload, "geometry", "condition_number", required=True),
    )


"common(x_1,...,x_m) = x if all equal else 'MIXED'"
def _common_or_mixed(values: Sequence[Any]) -> Any:
    rendered = [repr(v) for v in values]
    if not rendered:
        return "(missing)"
    if all(token == rendered[0] for token in rendered[1:]):
        return values[0]
    return "MIXED"


"logsafe(x) = max(|x|, ε)"
def _positive_floor(values: np.ndarray, eps: float = 1.0e-18) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.maximum(np.abs(arr), float(eps))


"summary_row(entry) = headline_metrics(entry)"
def _summary_row(entry: ReportEntry) -> list[str]:
    summary = entry.payload.get("summary", {}) if isinstance(entry.payload.get("summary", {}), Mapping) else {}
    return [
        entry.label,
        str(summary.get("runtime_parameter_count", "?")),
        f"{float(entry.fidelity[-1]):.6f}",
        f"{float(np.min(entry.fidelity)):.6f}",
        f"{float(np.max(np.abs(entry.abs_energy_error))):.3e}",
        f"{float(np.max(entry.rho_miss)):.3f}",
        f"{float(np.max(np.abs(entry.theta_dot_l2))):.3e}",
        f"{float(np.max(np.abs(entry.condition_number))):.3e}",
    ]


"physics_signature = (L,t,U,dv,omega0,g_ep,n_ph_max,boundary,ordering)"
def _physics_signature(entry: ReportEntry) -> tuple[Any, ...]:
    settings = entry.source_settings
    return (
        settings.get("L"),
        settings.get("t"),
        settings.get("u"),
        settings.get("dv"),
        settings.get("omega0"),
        settings.get("g_ep"),
        settings.get("n_ph_max"),
        settings.get("boundary"),
        settings.get("ordering"),
    )


"compatible(entries) = same(physics, drive_profile, projection_config, drive_enabled)"
def _validate_report_compatibility(entries: Sequence[ReportEntry]) -> None:
    if not entries:
        raise ValueError("At least one report entry is required.")
    physics = {_physics_signature(entry) for entry in entries}
    drive_signatures = {
        json.dumps(entry.payload.get("drive_profile", {}), sort_keys=True, default=str)
        for entry in entries
    }
    projection_signatures = {
        json.dumps(entry.payload.get("projection_config", {}), sort_keys=True, default=str)
        for entry in entries
    }
    drive_enabled = {
        bool((entry.payload.get("manifest", {}) if isinstance(entry.payload.get("manifest", {}), Mapping) else {}).get("drive_enabled", False))
        for entry in entries
    }
    if len(physics) != 1 or len(drive_signatures) != 1 or len(projection_signatures) != 1 or len(drive_enabled) != 1:
        raise ValueError(
            "All report inputs must share the same physics, drive profile, projection config, and drive-enabled flag."
        )


"manifest_extra = common_physics ∪ run_controls ∪ drive_controls"
def _manifest_extra(entries: Sequence[ReportEntry]) -> dict[str, Any]:
    first = entries[0]
    first_drive = first.payload.get("drive_profile", {}) if isinstance(first.payload.get("drive_profile", {}), Mapping) else {}
    first_manifest = first.payload.get("manifest", {}) if isinstance(first.payload.get("manifest", {}), Mapping) else {}
    return {
        "L": _common_or_mixed([entry.source_settings.get("L") for entry in entries]),
        "omega0": _common_or_mixed([entry.source_settings.get("omega0") for entry in entries]),
        "g_ep": _common_or_mixed([entry.source_settings.get("g_ep") for entry in entries]),
        "n_ph_max": _common_or_mixed([entry.source_settings.get("n_ph_max") for entry in entries]),
        "boundary": _common_or_mixed([entry.source_settings.get("boundary") for entry in entries]),
        "ordering": _common_or_mixed([entry.source_settings.get("ordering") for entry in entries]),
        "t_final": _common_or_mixed([
            (entry.payload.get("run_config", {}) if isinstance(entry.payload.get("run_config", {}), Mapping) else {}).get("t_final")
            for entry in entries
        ]),
        "num_times": _common_or_mixed([
            (entry.payload.get("run_config", {}) if isinstance(entry.payload.get("run_config", {}), Mapping) else {}).get("num_times")
            for entry in entries
        ]),
        "geometry_backend": _common_or_mixed([
            (entry.payload.get("manifest", {}) if isinstance(entry.payload.get("manifest", {}), Mapping) else {}).get("geometry_backend")
            for entry in entries
        ]),
        "reference_audit": _common_or_mixed([
            (entry.payload.get("manifest", {}) if isinstance(entry.payload.get("manifest", {}), Mapping) else {}).get("reference_audit_method")
            for entry in entries
        ]),
        "drive_pattern": first_drive.get("pattern"),
        "drive_A": first_drive.get("A"),
        "drive_omega": first_drive.get("omega"),
        "drive_tbar": first_drive.get("tbar"),
        "drive_phi": first_drive.get("phi"),
        "drive_t0": first_drive.get("t0"),
        "drive_sampling": first_drive.get("time_sampling"),
        "noise_mode": first_manifest.get("noise_mode"),
        "projection_integrator": (
            (first.payload.get("projection_config", {}) if isinstance(first.payload.get("projection_config", {}), Mapping) else {}).get("integrator")
        ),
        "source_artifacts": "; ".join(str(entry.source_artifact_path) for entry in entries),
    }


"overview_lines = human_readable_provenance(entries)"
def _overview_lines(entries: Sequence[ReportEntry]) -> list[str]:
    driven = bool((entries[0].payload.get("manifest", {}) if isinstance(entries[0].payload.get("manifest", {}), Mapping) else {}).get("drive_enabled", False))
    reference_label = "exact driven reference" if driven else "exact reference"
    lines = [
        "Fixed-manifold HH time-dynamics report",
        "",
        "How to read this PDF",
        f"- Each run is compared against its own {reference_label} built from the same initial prepared state.",
        "- Fidelity close to 1 is good.",
        "- |ΔE_total| is the absolute energy mismatch between the variational state and the exact reference at that time.",
        "- rho_miss measures how much desired motion sits outside the chosen manifold: near 0 is good, near 1 is bad.",
        "- theta_dot_l2 is the size of the McLachlan parameter update; near 0 means the manifold is barely moving.",
        "- Condition number measures geometry ill-conditioning; larger means shot-noise/QPU use will be harder.",
        "",
        "Same physics across runs: True",
        "- Overlay pages are visually useful, but not a strict apples-to-apples initial-state comparison.",
        "",
        "Input measured-run JSONs",
    ]
    for entry in entries:
        loader = entry.payload.get("loader", {}) if isinstance(entry.payload.get("loader", {}), Mapping) else {}
        lines.append(
            f"- {entry.label}: {entry.json_path} | loader={loader.get('loader_mode')} | source={entry.source_artifact_path}"
        )
    return lines


"plot_drive(pdf, times, drive_profile, L) = waveform_pages"
def _render_drive_page(pdf: Any, entries: Sequence[ReportEntry]) -> None:
    require_matplotlib()
    plt = get_plt()
    first = entries[0]
    drive_profile = first.payload.get("drive_profile", {}) if isinstance(first.payload.get("drive_profile", {}), Mapping) else {}
    if not drive_profile or not bool((first.payload.get("manifest", {}) if isinstance(first.payload.get("manifest", {}), Mapping) else {}).get("drive_enabled", False)):
        return
    L = int(first.source_settings.get("L", 0))
    weights = default_spatial_weights(
        L,
        mode=str(drive_profile.get("pattern", "staggered")),
        custom=drive_profile.get("custom_weights"),
    )
    waveform = evaluate_drive_waveform(first.times, drive_profile, float(drive_profile.get("A", 0.0)))
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), sharex=True)
    axes[0].plot(first.times, waveform, color="#1f77b4", linewidth=2.0)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_title("Driven scalar waveform v(t)")
    axes[0].set_ylabel("v(t)")
    axes[0].grid(True, alpha=0.25)
    for site_idx, weight in enumerate(weights.tolist()):
        axes[1].plot(
            first.times,
            float(weight) * waveform,
            linewidth=1.8,
            label=f"site {site_idx} (weight={float(weight):+.1f})",
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_title("Site-resolved onsite-density drive s_j v(t)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Potential")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


"overlay(entries) = multi_run_time_series"
def _render_overlay_page(pdf: Any, entries: Sequence[ReportEntry]) -> None:
    require_matplotlib()
    plt = get_plt()
    driven = bool((entries[0].payload.get("manifest", {}) if isinstance(entries[0].payload.get("manifest", {}), Mapping) else {}).get("drive_enabled", False))
    reference_label = "exact driven reference" if driven else "exact reference"
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
    ax_fid, ax_err, ax_rho, ax_theta = axes.ravel()
    for entry in entries:
        ax_fid.plot(entry.times, entry.fidelity, linewidth=2.0, label=entry.label)
        ax_err.semilogy(entry.times, _positive_floor(entry.abs_energy_error), linewidth=2.0, label=entry.label)
        ax_rho.plot(entry.times, entry.rho_miss, linewidth=2.0, label=entry.label)
        ax_theta.semilogy(entry.times, _positive_floor(entry.theta_dot_l2), linewidth=2.0, label=entry.label)
    ax_fid.set_title(f"Fidelity vs {reference_label}")
    ax_fid.set_ylabel("Fidelity")
    ax_fid.grid(True, alpha=0.25)
    ax_err.set_title(f"Absolute total-energy error vs {reference_label}")
    ax_err.set_ylabel("|ΔE_total|")
    ax_err.grid(True, alpha=0.25)
    ax_rho.set_title("Projected-miss ratio")
    ax_rho.set_xlabel("Time")
    ax_rho.set_ylabel("rho_miss")
    ax_rho.set_ylim(-0.02, 1.05)
    ax_rho.grid(True, alpha=0.25)
    ax_theta.set_title("Parameter-speed norm")
    ax_theta.set_xlabel("Time")
    ax_theta.set_ylabel("theta_dot_l2")
    ax_theta.grid(True, alpha=0.25)
    handles, labels = ax_fid.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(entries)), fontsize=9)
    title_prefix = "Driven" if driven else "Static"
    fig.suptitle(
        f"{title_prefix} fixed-manifold McLachlan overlay\nvisual comparison only; prepared-state anchors differ across manifolds",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


"run_page(entry) = six_panel_trajectory_summary(entry)"
def _render_run_page(pdf: Any, entry: ReportEntry) -> None:
    require_matplotlib()
    plt = get_plt()
    driven = bool((entry.payload.get("manifest", {}) if isinstance(entry.payload.get("manifest", {}), Mapping) else {}).get("drive_enabled", False))
    reference_label = "exact driven reference" if driven else "exact reference"
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 8.5), sharex=True)
    ax_fid, ax_energy, ax_err, ax_rho, ax_theta, ax_cond = axes.ravel()

    ax_fid.plot(entry.times, entry.fidelity, color="#1f77b4", linewidth=2.0)
    ax_fid.set_title(f"Fidelity vs {reference_label}")
    ax_fid.set_ylabel("Fidelity")
    ax_fid.grid(True, alpha=0.25)

    ax_energy.plot(entry.times, entry.energy_ansatz, color="#1f77b4", linewidth=2.0, label="ansatz")
    ax_energy.plot(entry.times, entry.energy_reference, color="#d62728", linewidth=1.8, linestyle="--", label="exact ref")
    ax_energy.set_title("Energy traces")
    ax_energy.set_ylabel("Energy")
    ax_energy.grid(True, alpha=0.25)
    ax_energy.legend(fontsize=8, loc="best")

    ax_err.semilogy(entry.times, _positive_floor(entry.abs_energy_error), color="#9467bd", linewidth=2.0)
    ax_err.set_title("Absolute total-energy error")
    ax_err.set_ylabel("|ΔE_total|")
    ax_err.grid(True, alpha=0.25)

    ax_rho.plot(entry.times, entry.rho_miss, color="#2ca02c", linewidth=2.0)
    ax_rho.set_title("Projected-miss ratio")
    ax_rho.set_ylabel("rho_miss")
    ax_rho.set_ylim(-0.02, 1.05)
    ax_rho.grid(True, alpha=0.25)

    ax_theta.semilogy(entry.times, _positive_floor(entry.theta_dot_l2), color="#ff7f0e", linewidth=2.0)
    ax_theta.set_title("Parameter-speed norm")
    ax_theta.set_xlabel("Time")
    ax_theta.set_ylabel("theta_dot_l2")
    ax_theta.grid(True, alpha=0.25)

    ax_cond.semilogy(entry.times, _positive_floor(entry.condition_number), color="#8c564b", linewidth=2.0)
    ax_cond.set_title("Geometry condition number")
    ax_cond.set_xlabel("Time")
    ax_cond.set_ylabel("cond(G)")
    ax_cond.grid(True, alpha=0.25)

    loader = entry.payload.get("loader", {}) if isinstance(entry.payload.get("loader", {}), Mapping) else {}
    summary = entry.payload.get("summary", {}) if isinstance(entry.payload.get("summary", {}), Mapping) else {}
    fig.suptitle(
        (
            f"{entry.label} | loader={loader.get('loader_mode')} | params={summary.get('runtime_parameter_count')} "
            f"| source={entry.source_artifact_path.name}"
        ),
        fontsize=13,
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


"write_pdf(entries, out) = manifest + overview + tables + plots + command"
def write_fixed_manifold_dynamics_pdf(
    input_jsons: Sequence[str | Path],
    output_pdf: str | Path,
    *,
    run_command: str | None = None,
) -> Path:
    require_matplotlib()
    entries = [load_report_entry(path) for path in input_jsons]
    _validate_report_compatibility(entries)
    if not entries:
        raise ValueError("At least one input JSON is required.")
    output_path = Path(output_pdf).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first = entries[0]
    first_settings = first.source_settings
    ansatz_label = ", ".join(entry.label for entry in entries)
    manifest_extra = _manifest_extra(entries)
    PdfPages = get_PdfPages()
    plt = get_plt()
    with PdfPages(str(output_path)) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein",
            ansatz=f"fixed-manifold McLachlan: {ansatz_label}",
            drive_enabled=bool((first.payload.get("manifest", {}) if isinstance(first.payload.get("manifest", {}), Mapping) else {}).get("drive_enabled", False)),
            t=float(first_settings.get("t", 0.0)),
            U=float(first_settings.get("u", 0.0)),
            dv=float(first_settings.get("dv", 0.0)),
            extra=manifest_extra,
            command=run_command or current_command_string(),
        )
        render_text_page(pdf, _overview_lines(entries), fontsize=10, line_spacing=0.03)

        fig, ax = plt.subplots(figsize=(11.0, 8.5))
        render_compact_table(
            ax,
            title="Headline time-dynamics metrics",
            col_labels=[
                "Run",
                "Params",
                "Final fidelity",
                "Min fidelity",
                "Max |ΔE_total|",
                "Max rho_miss",
                "Max theta_dot",
                "Max cond(G)",
            ],
            rows=[_summary_row(entry) for entry in entries],
            fontsize=8,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        _render_drive_page(pdf, entries)
        if len(entries) > 1:
            _render_overlay_page(pdf, entries)
        for entry in entries:
            _render_run_page(pdf, entry)
        render_command_page(
            pdf,
            run_command or current_command_string(),
            script_name="pipelines/reporting/hh_fixed_manifold_dynamics_report.py",
            extra_header_lines=[
                f"output_pdf: {output_path}",
                *[f"input_json[{idx}]: {entry.json_path}" for idx, entry in enumerate(entries)],
            ],
        )
    return output_path


"argv -> parsed_cli_config"
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a PDF report for fixed-manifold HH time-dynamics JSON artifacts.",
    )
    parser.add_argument(
        "--input-json",
        action="append",
        required=True,
        help="Measured fixed-manifold dynamics JSON artifact. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--output-pdf",
        required=True,
        help="Output PDF path.",
    )
    return parser.parse_args(argv)


"main = write_pdf(parse_args(argv))"
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    write_fixed_manifold_dynamics_pdf(
        input_jsons=[str(x) for x in args.input_json],
        output_pdf=str(args.output_pdf),
        run_command=current_command_string(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

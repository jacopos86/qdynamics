#!/usr/bin/env python3
"""3D energy-visualization prototypes for hardcoded Hubbard outputs.

This script is intentionally isolated from production pipeline code.
It reads existing JSON artifacts and renders three distinct 3D views:
1) grouped bars over time,
2) dual surfaces over time,
3) waterfall time-slice stacks.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed to register 3D projection


ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = ROOT / "Fermi-Hamil-JW-VQE-TROTTER-PIPELINE"

DEFAULT_JSON_PATHS = [
    PIPELINE_ROOT / "artifacts" / "json" / "run_1_L3_vt_t1.0_U4.0_S128.json",
    PIPELINE_ROOT / "artifacts" / "json" / "run_2_L2_vt_t1.0_U4.0_S64.json",
]
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "artifacts" / "testing3d"


@dataclass(frozen=True)
class EnergyDataset:
    source_path: Path
    L: int
    times: np.ndarray
    e_total_exact: np.ndarray
    e_total_trotter: np.ndarray
    e_delta_trot_minus_exact: np.ndarray
    vqe_energy: float
    vqe_t0_delta: float
    vqe_t0_pass: bool
    vqe_t0_tol: float


def _fmt_num(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.3e}"


def _parse_json_paths(raw: str | None) -> list[Path]:
    if raw is None or str(raw).strip() == "":
        return list(DEFAULT_JSON_PATHS)
    out: list[Path] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if t:
            out.append(Path(t))
    return out


def _downsample_indices(n: int, max_points: int | None) -> np.ndarray:
    if max_points is None or max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)
    idx = np.linspace(0, n - 1, int(max_points), dtype=int)
    return np.unique(idx)


def _extract_series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals: list[float] = []
    for i, row in enumerate(rows):
        if key not in row:
            raise KeyError(f"Missing trajectory key '{key}' at row {i}")
        vals.append(float(row[key]))
    arr = np.asarray(vals, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values found in '{key}'")
    return arr


def _load_dataset(path: Path, *, vqe_t0_tol: float, max_time_points: int | None) -> EnergyDataset:
    data = json.loads(path.read_text(encoding="utf-8"))
    settings = data.get("settings", {})
    rows = data.get("trajectory", [])
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError(f"Empty or missing trajectory in {path}")

    L = int(settings.get("L"))
    times = _extract_series(rows, "time")
    e_exact = _extract_series(rows, "energy_total_exact")
    e_trot = _extract_series(rows, "energy_total_trotter")

    if not np.all(np.diff(times) >= -1e-12):
        raise ValueError(f"Time grid is not nondecreasing in {path}")

    keep = _downsample_indices(int(times.size), max_time_points)
    times = times[keep]
    e_exact = e_exact[keep]
    e_trot = e_trot[keep]

    e_delta = e_trot - e_exact

    vqe_block = data.get("vqe", {})
    vqe_raw = vqe_block.get("energy") if isinstance(vqe_block, dict) else None
    vqe_energy = float(vqe_raw) if vqe_raw is not None else float("nan")
    if not np.isfinite(vqe_energy):
        vqe_energy = float("nan")

    if np.isfinite(vqe_energy):
        vqe_t0_delta = abs(float(vqe_energy) - float(e_trot[0]))
        vqe_t0_pass = bool(vqe_t0_delta <= float(vqe_t0_tol))
    else:
        vqe_t0_delta = float("nan")
        vqe_t0_pass = False

    return EnergyDataset(
        source_path=path,
        L=L,
        times=times,
        e_total_exact=e_exact,
        e_total_trotter=e_trot,
        e_delta_trot_minus_exact=e_delta,
        vqe_energy=vqe_energy,
        vqe_t0_delta=vqe_t0_delta,
        vqe_t0_pass=vqe_t0_pass,
        vqe_t0_tol=float(vqe_t0_tol),
    )


def _common_title(ds: EnergyDataset, view_name: str) -> str:
    status = "PASS" if ds.vqe_t0_pass else "FAIL"
    return (
        f"L={ds.L} | {view_name}\n"
        f"Energy: H(t) total | VQE@t0 check: {status} "
        f"(|Δ|={_fmt_num(ds.vqe_t0_delta)}, tol={ds.vqe_t0_tol:.1e})"
    )


def _add_vqe_anchor(ax: Any, ds: EnergyDataset, *, y_value: float) -> None:
    if not np.isfinite(ds.vqe_energy):
        return
    marker = "*" if ds.vqe_t0_pass else "X"
    color = "#111111" if ds.vqe_t0_pass else "#d62728"
    label = "VQE @ t=0 (matched)" if ds.vqe_t0_pass else "VQE @ t=0 (mismatch)"
    ax.scatter(
        [float(ds.times[0])],
        [float(y_value)],
        [float(ds.vqe_energy)],
        marker=marker,
        s=100,
        c=color,
        label=label,
        depthshade=False,
        edgecolors="white",
        linewidths=0.6,
    )


def _plot_view1_bars(ds: EnergyDataset) -> plt.Figure:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    x = ds.times
    n = int(x.size)
    dt = np.diff(x)
    dx = float(np.median(dt)) * 0.75 if dt.size > 0 else 0.25
    dx = max(dx, 1e-3)
    y_exact = 0.0
    y_trot = 0.36
    dy = 0.14

    series = [
        ("Exact total", y_exact, ds.e_total_exact, "#1f77b4"),
        ("Trotter total", y_trot, ds.e_total_trotter, "#ff7f0e"),
    ]

    all_vals = np.concatenate([ds.e_total_exact, ds.e_total_trotter])
    z_min = float(np.min(all_vals))
    z_max = float(np.max(all_vals))
    span = max(z_max - z_min, 1e-9)
    z_base = z_min - 0.08 * span

    for _name, y_lane, vals, color in series:
        x0 = x - dx * 0.5
        y0 = np.full(n, float(y_lane) - dy * 0.5, dtype=float)
        z0 = np.full(n, z_base, dtype=float)
        dz = vals - z_base
        ax.bar3d(x0, y0, z0, dx, dy, dz, color=color, shade=True, alpha=0.88)

    _add_vqe_anchor(ax, ds, y_value=y_trot + 0.10)

    ax.set_ylim(y_exact - 0.10, y_trot + 0.18)
    ax.set_yticks([y_exact, y_trot])
    ax.set_yticklabels(["Exact", "Trotter"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Series")
    ax.set_zlabel("Energy")
    ax.set_title(_common_title(ds, "3D Grouped Bars"), pad=14)
    ax.view_init(elev=28, azim=-57)

    handles = [Patch(facecolor=c, edgecolor="black", label=nm) for nm, _y, _vals, c in series]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_view2_surfaces(ds: EnergyDataset) -> plt.Figure:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    y_exact = 0.0
    y_trot = 0.36
    half_width = 0.05
    x = ds.times
    z_exact = ds.e_total_exact
    z_trot = ds.e_total_trotter

    for yi, vals, color, label in [
        (y_exact, z_exact, "#1f77b4", "Exact surface"),
        (y_trot, z_trot, "#ff7f0e", "Trotter surface"),
    ]:
        x_grid = np.vstack([x, x])
        y_grid = np.vstack([np.full_like(x, yi - half_width), np.full_like(x, yi + half_width)])
        z_grid = np.vstack([vals, vals])
        ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            color=color,
            alpha=0.75,
            linewidth=0.0,
            antialiased=True,
        )
        ax.plot(x, np.full_like(x, yi), vals, color=color, linewidth=1.5, label=label)

    _add_vqe_anchor(ax, ds, y_value=y_trot + 0.03)

    ax.set_ylim(y_exact - 0.10, y_trot + 0.12)
    ax.set_yticks([y_exact, y_trot])
    ax.set_yticklabels(["Exact", "Trotter"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Series")
    ax.set_zlabel("Energy")
    ax.set_title(_common_title(ds, "3D Dual Surface"), pad=14)
    ax.view_init(elev=24, azim=-36)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_view3_waterfall(ds: EnergyDataset) -> plt.Figure:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    y_exact = 0.0
    y_trot = 0.36
    x = ds.times
    n = int(x.size)
    n_slices = min(n, 90)
    slice_idx = np.unique(np.linspace(0, n - 1, n_slices, dtype=int))
    norm = Normalize(vmin=float(x[slice_idx[0]]), vmax=float(x[slice_idx[-1]]))
    cmap = cm.plasma

    for i in slice_idx:
        t = float(x[i])
        y_line = np.array([y_exact, y_trot], dtype=float)
        z_line = np.array(
            [
                float(ds.e_total_exact[i]),
                float(ds.e_total_trotter[i]),
            ],
            dtype=float,
        )
        x_line = np.full(2, t, dtype=float)
        ax.plot(x_line, y_line, z_line, color=cmap(norm(t)), linewidth=0.9, alpha=0.85)

    ax.plot(x, np.full_like(x, y_exact), ds.e_total_exact, color="#1f77b4", linewidth=2.1, label="Exact lane")
    ax.plot(x, np.full_like(x, y_trot), ds.e_total_trotter, color="#ff7f0e", linewidth=2.1, label="Trotter lane")
    _add_vqe_anchor(ax, ds, y_value=y_trot + 0.03)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.08)
    cbar.set_label("Time (slice color)")

    ax.set_ylim(y_exact - 0.10, y_trot + 0.12)
    ax.set_yticks([y_exact, y_trot])
    ax.set_yticklabels(["Exact", "Trotter"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Series")
    ax.set_zlabel("Energy")
    ax.set_title(_common_title(ds, "3D Waterfall Time-Slices"), pad=14)
    ax.view_init(elev=30, azim=33)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def _write_outputs(ds: EnergyDataset, output_dir: Path, *, dpi: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"testing3d_L{ds.L}"
    png1 = output_dir / f"{stem}_view1_bars.png"
    png2 = output_dir / f"{stem}_view2_surface.png"
    png3 = output_dir / f"{stem}_view3_waterfall.png"
    pdf_all = output_dir / f"{stem}_all_views.pdf"
    summary_path = output_dir / f"{stem}_summary.json"

    fig1 = _plot_view1_bars(ds)
    fig2 = _plot_view2_surfaces(ds)
    fig3 = _plot_view3_waterfall(ds)

    fig1.savefig(png1, dpi=int(dpi), bbox_inches="tight")
    fig2.savefig(png2, dpi=int(dpi), bbox_inches="tight")
    fig3.savefig(png3, dpi=int(dpi), bbox_inches="tight")

    with PdfPages(str(pdf_all)) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

    summary = {
        "source_json": str(ds.source_path),
        "L": int(ds.L),
        "energy_series": "energy_total_exact / energy_total_trotter",
        "num_time_points_plotted": int(ds.times.size),
        "time_start": float(ds.times[0]),
        "time_end": float(ds.times[-1]),
        "vqe_t0_check": {
            "passed": bool(ds.vqe_t0_pass),
            "delta_abs": float(ds.vqe_t0_delta) if np.isfinite(ds.vqe_t0_delta) else None,
            "tolerance": float(ds.vqe_t0_tol),
            "vqe_energy": float(ds.vqe_energy) if np.isfinite(ds.vqe_energy) else None,
            "energy_total_trotter_t0": float(ds.e_total_trotter[0]),
        },
        "outputs": {
            "png_view1_bars": str(png1),
            "png_view2_surface": str(png2),
            "png_view3_waterfall": str(png3),
            "pdf_all_views": str(pdf_all),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D prototype charts for Hubbard energy evolution.")
    parser.add_argument(
        "--json-paths",
        type=str,
        default=",".join(str(p) for p in DEFAULT_JSON_PATHS),
        help="Comma-separated JSON paths. Default uses run_1 and run_2 hardcoded outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for 3D prototype outputs.",
    )
    parser.add_argument(
        "--vqe-t0-tol",
        type=float,
        default=1e-8,
        help="Tolerance for |VQE - energy_total_trotter(t=0)| consistency test.",
    )
    parser.add_argument(
        "--max-time-points",
        type=int,
        default=None,
        help="Optional downsampling cap for plotted time points.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG export DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_paths = _parse_json_paths(args.json_paths)
    if not json_paths:
        raise ValueError("No input JSON paths provided.")

    datasets: list[EnergyDataset] = []
    for raw in json_paths:
        p = raw if raw.is_absolute() else (ROOT / raw)
        if not p.exists():
            raise FileNotFoundError(f"Input JSON not found: {p}")
        ds = _load_dataset(
            p,
            vqe_t0_tol=float(args.vqe_t0_tol),
            max_time_points=args.max_time_points,
        )
        datasets.append(ds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, Any]] = []
    for ds in datasets:
        summary = _write_outputs(ds, args.output_dir, dpi=int(args.dpi))
        all_summaries.append(summary)
        print(
            f"L={ds.L}: wrote 3 PNG + 1 PDF + summary "
            f"(vqe_t0_pass={ds.vqe_t0_pass}, delta={_fmt_num(ds.vqe_t0_delta)})"
        )

    index_path = args.output_dir / "testing3d_index.json"
    index_payload = {
        "generated_by": str(Path(__file__).name),
        "num_datasets": len(all_summaries),
        "datasets": all_summaries,
    }
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    print(f"Wrote index: {index_path}")


if __name__ == "__main__":
    main()

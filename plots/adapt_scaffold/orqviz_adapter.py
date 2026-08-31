"""Optional isolated adapter for orqviz-based landscape overlays."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from docs.reports.pdf_utils import get_plt, require_matplotlib
from plots.adapt_scaffold.landscape import FinalScaffoldEnergyModel, ScaffoldTrajectoryDataset


def _scan_directions(prefix_runtime_points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    runtime_dim = int(prefix_runtime_points.shape[1])
    if runtime_dim < 2:
        return None
    centered = prefix_runtime_points - np.mean(prefix_runtime_points, axis=0, keepdims=True)
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    if vt.shape[0] < 2 or np.sum(singular_values > 1.0e-12) < 2:
        return None
    return np.asarray(vt[0], dtype=float), np.asarray(vt[1], dtype=float)


def try_write_orqviz_overlay(
    *,
    energy_model: FinalScaffoldEnergyModel,
    dataset: ScaffoldTrajectoryDataset,
    output_png: Path,
    output_json: Path,
) -> dict[str, Any]:
    try:
        scans_mod = importlib.import_module("orqviz.scans")
    except ModuleNotFoundError:
        return {"status": "skipped_optional_dependency", "reason": "orqviz_not_installed"}

    directions = _scan_directions(dataset.prefix_runtime_points)
    if directions is None:
        return {"status": "skipped_prerequisite", "reason": "insufficient_runtime_rank_for_2d_scan"}
    direction_x, direction_y = directions
    center = np.asarray(dataset.prefix_runtime_points[-1], dtype=float)

    def _loss_function(params: np.ndarray) -> float:
        return float(energy_model.energy(np.asarray(params, dtype=float)))

    try:
        scan_result = scans_mod.perform_2D_scan(
            center,
            _loss_function,
            direction_x=direction_x,
            direction_y=direction_y,
            n_steps_x=25,
            n_steps_y=25,
        )
    except Exception as exc:  # pragma: no cover - best effort for optional dep
        return {"status": "error", "reason": f"orqviz_scan_failed:{type(exc).__name__}:{exc}"}

    require_matplotlib()
    plt = get_plt()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    fig = None
    ax = None
    try:
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        try:
            scans_mod.plot_2D_scan_result(scan_result, ax=ax)
        except TypeError:
            plt.close(fig)
            plotted = scans_mod.plot_2D_scan_result(scan_result)
            if hasattr(plotted, "savefig"):
                fig = plotted
                ax = plotted.axes[0] if getattr(plotted, "axes", None) else None
            elif isinstance(plotted, tuple) and plotted and hasattr(plotted[0], "savefig"):
                fig = plotted[0]
                ax = plotted[1] if len(plotted) > 1 else None
            else:
                return {"status": "error", "reason": "orqviz_plot_return_shape_unsupported"}

        rel = dataset.prefix_runtime_points - center[None, :]
        proj_x = rel @ direction_x
        proj_y = rel @ direction_y
        if ax is None:
            ax = fig.axes[0]
        ax.plot(proj_x, proj_y, color="white", linewidth=1.6, marker="o", markersize=4.5, zorder=5)
        ax.set_title("Final scaffold trajectory overlay on optional orqviz scan")
        fig.tight_layout()
        fig.savefig(output_png, dpi=180, bbox_inches="tight")
        output_json.write_text(
            json.dumps(
                {
                    "status": "written",
                    "center": center.tolist(),
                    "trajectory_overlay_points": [
                        {"point_index": int(i), "scan_x": float(x), "scan_y": float(y)}
                        for i, (x, y) in enumerate(zip(proj_x, proj_y, strict=False))
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"status": "written", "path_png": str(output_png), "path_json": str(output_json)}
    finally:  # pragma: no branch
        if fig is not None:
            plt.close(fig)


__all__ = ["try_write_orqviz_overlay"]

